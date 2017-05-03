/*******************************************************************************
* Copyright 2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx512_mic_1x1_convolution.hpp"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <bool with_relu>
void _jit_avx512_mic_1x1_convolution_fwd_t<with_relu>::execute_forward()
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const auto &jcp = kernel_->jcp;

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    const int stride_h = conf_.cdesc()->strides[0];
    const int stride_w = conf_.cdesc()->strides[1];
    const int pad_t = conf_.cdesc()->padding[0][0];
    const int pad_l = conf_.cdesc()->padding[0][1];

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };
    auto ker = [&](const int ithr, const int nthr) {
        // TODO (Roma): remove this restriction
        assert(jcp.stride_w == 1 && jcp.stride_h == 1);

        jit_1x1_conv_call_s p = {};
        rtus_driver_f32_t<avx512_mic>::call_params_t rp = {};

        const int nb_oc = jcp.nb_load;
        const int nb_ic = jcp.nb_reduce;
        const int nb_ic_blocking = jcp.nb_reduce_blocking;
        const int os_block = jcp.bcast_block;

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int iwork = start;
        while (iwork < end) {
            int n{0}, g{0}, osb{0};
            nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
                             jcp.nb_bcast);

            int bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                    jcp.nb_bcast_blocking_max);
            bcast_step = nstl::min(bcast_step, end - iwork);

            const int os = osb * os_block;
            const int oh = os / jcp.ow;
            const int ow = os % jcp.ow;

            const int ih = nstl::max(oh * stride_h - pad_t, 0);
            const int iw = nstl::max(ow * stride_w - pad_l, 0);
            rp.iw_start = iw;

            p.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);
            rp.os = p.bcast_dim;

            int ocb = 0;
            while (ocb < jcp.nb_load) {
                const int load_step = step(jcp.nb_load_blocking,
                        jcp.nb_load - ocb, jcp.nb_load_blocking_max);

                const int _ocb = g * nb_oc + ocb;
                p.load_dim = this_block_size(ocb * jcp.oc_block, jcp.oc,
                        load_step * jcp.oc_block);

                const size_t dst_off = dst_d.blk_off(n, _ocb, oh, ow);
                p.output_data = &dst[dst_off];

                p.bias_data = &bias[_ocb * jcp.oc_block];

                for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                    p.reduce_pos_flag = 0
                        | (icb == 0
                                ?
                                jit_avx512_mic_1x1_conv_kernel_f32::REDUCE_FLAG_FIRST
                                : 0)
                        | (icb + nb_ic_blocking >= nb_ic
                                ?
                                jit_avx512_mic_1x1_conv_kernel_f32::REDUCE_FLAG_LAST
                                : 0);

                    p.reduce_dim = this_block_size(icb * jcp.ic_block, jcp.ic,
                            nb_ic_blocking * jcp.ic_block);
                    rp.icb = p.reduce_dim / jcp.reduce_block;

                    p.load_data = &weights[conf_.with_groups()
                        ? weights_d.blk_off(g, ocb, icb)
                        : weights_d.blk_off(ocb, icb)];

                    const int _icb = g * nb_ic + icb;
                    if (conf_.rtus_.reduce_src_) {
                        rp.ws = scratch_ + ithr * ws_per_thread_
                            + _icb * jcp.is * jcp.ic_block;

                        if (ocb == 0) {
                            rp.src = src + src_d.blk_off(n, _icb, ih, iw);
                            rtus_driver_->ker_(&rp);
                        }

                        p.bcast_data = rp.ws;
                    } else
                        p.bcast_data = src + src_d.blk_off(n, _icb, ih, iw);

                    kernel_->jit_ker(&p);
                }

                ocb += load_step;
            }

            iwork += bcast_step;
        }
    };

#   pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template void _jit_avx512_mic_1x1_convolution_fwd_t<true>::execute_forward();
template void _jit_avx512_mic_1x1_convolution_fwd_t<false>::execute_forward();

void jit_avx512_mic_1x1_convolution_bwd_data_t::execute_backward_data()
{
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());

    const auto &jcp = kernel_->jcp;

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    const int stride_h = conf_.desc()->strides[0];
    const int stride_w = conf_.desc()->strides[1];
    const int pad_t = conf_.desc()->padding[0][0];
    const int pad_l = conf_.desc()->padding[0][1];

    const int nb_ic = jcp.nb_load;
    const int nb_oc = jcp.nb_reduce;
    const int os_block = jcp.bcast_block;
    const int nb_oc_blocking = jcp.nb_reduce_blocking;

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto ker = [&](const int ithr, const int nthr) {
        jit_1x1_conv_call_s p = {};
        rtus_driver_f32_t<avx512_mic>::call_params_t rp = {};

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int load_step = 0;
        for (int icb = 0; icb < jcp.nb_load; icb += load_step) {
            load_step = step(jcp.nb_load_blocking, jcp.nb_load - icb,
                    jcp.nb_load_blocking_max);

            p.load_dim = this_block_size(icb * jcp.ic_block, jcp.ic,
                    load_step * jcp.ic_block);
            rp.icb = p.load_dim / jcp.ic_block;

            int bcast_step;
            for (int iwork = start; iwork < end; iwork += bcast_step) {
                int n{0}, g{0}, osb{0};
                nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
                        jcp.nb_bcast);

                bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                        jcp.nb_bcast_blocking_max);
                bcast_step = nstl::min(bcast_step, end - iwork);

                const int os = osb * os_block;
                p.bcast_dim = this_block_size(os, jcp.os,
                        bcast_step * os_block);
                rp.os = p.bcast_dim;

                const int oh = os / jcp.ow;
                const int ow = os % jcp.ow;
                const int ih = nstl::max(oh * stride_h - pad_t, 0);
                const int iw = nstl::max(ow * stride_w - pad_l, 0);
                rp.iw_start = iw;

                const int _icb = g * nb_ic + icb;
                rp.src = diff_src + diff_src_d.blk_off(n, _icb, ih, iw);

                if (conf_.rtus_.reduce_src_) {
                    rp.ws = scratch_ + ithr * ws_per_thread_;
                    p.output_data = rp.ws;
                } else
                    p.output_data = rp.src;

                for (int ocb = 0; ocb < jcp.nb_reduce;
                        ocb += jcp.nb_reduce_blocking) {
                    const int _ocb = g * nb_oc + ocb;
                    size_t diff_dst_off = diff_dst_d.blk_off(n, _ocb, oh, ow);
                    p.bcast_data = &diff_dst[diff_dst_off];

                    p.load_data = &weights[conf_.with_groups()
                        ? weights_d.blk_off(g, ocb, icb)
                        : weights_d.blk_off(ocb, icb)];

                    p.reduce_pos_flag = ocb == 0
                        ? jit_avx512_mic_1x1_conv_kernel_f32::REDUCE_FLAG_FIRST : 0;

                    p.reduce_dim = this_block_size(ocb * jcp.oc_block, jcp.oc,
                            nb_oc_blocking * jcp.oc_block);

                    kernel_->jit_ker(&p);
                }

                if (conf_.rtus_.reduce_src_)
                    rtus_driver_->ker_(&rp);
            }
        }
    };

#   pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

jit_avx512_mic_1x1_convolution_bwd_weights_t::jit_avx512_mic_1x1_convolution_bwd_weights_t(
        const pd_t *pd, const input_vector &inputs,
        const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), kernel_(nullptr)
    , rtus_driver_(nullptr), ws_per_thread_(0), scratch_(nullptr)
{
    kernel_ = new jit_avx512_mic_1x1_conv_kernel_f32(conf_.jcp_);

    const auto &jcp = kernel_->jcp;

    const int ic_block = jcp.bcast_block;
    const int nb_ic = jcp.nb_bcast;
    const int nb_ic_blocking = jcp.nb_bcast_blocking;
    const int bcast_work = utils::div_up(nb_ic, nb_ic_blocking);

    const int oc_block = jcp.load_block;
    const int nb_oc = jcp.nb_load;
    const int nb_oc_blocking = jcp.nb_load_blocking;
    const int load_work = utils::div_up(nb_oc, nb_oc_blocking);

    const int job_size
        = nb_oc_blocking * nb_ic_blocking * ic_block * oc_block;
    const int njobs_x = bcast_work;
    const int njobs_y = jcp.ngroups * load_work;

    const int simd_w = 16;
    const int max_threads = omp_get_max_threads();
    const size_t max_buffer_size = max_threads * job_size * simd_w;

    reducer_weights_ = new cpu_reducer_2d_t<data_type::f32>(
            reduce_balancer_t(max_threads, job_size, njobs_y * njobs_x,
                jcp.mb * jcp.reduce_dim, max_buffer_size),
            job_size / nb_oc_blocking, nb_oc_blocking,
            nb_ic * ic_block * oc_block, nb_oc, false);

    reducer_bias_ = !conf_.with_bias() ? nullptr
        : new cpu_reducer_t<data_type::f32>(reduce_balancer_t(max_threads,
                    oc_block, conf_.G() * conf_.OC() / oc_block,
                    conf_.MB(), max_buffer_size));

    init_rtus_driver_f32<avx512_mic>(this);
}
void jit_avx512_mic_1x1_convolution_bwd_weights_t::execute_backward_weights()
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    const auto &jcp = kernel_->jcp;

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    const int simd_w = 16;

    const int nb_ic = jcp.nb_bcast;
    const int nb_ic_blocking = jcp.nb_bcast_blocking;
    const int bcast_work = div_up(nb_ic, nb_ic_blocking);

    const int nb_oc = jcp.nb_load;
    const int nb_oc_blocking = jcp.nb_load_blocking;
    const int load_work = div_up(nb_oc, nb_oc_blocking);

    const int sp_dim = jcp.reduce_dim;
    const int mb_sp_work = jcp.mb * sp_dim;

    const int stride_h = conf_.desc()->strides[0];
    const int stride_w = conf_.desc()->strides[1];
    const int pad_t = conf_.desc()->padding[0][0];
    const int pad_l = conf_.desc()->padding[0][1];

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto oc_ic_sp_loop = [=](int sp_start, int sp_end, bool first_image,
            data_t *store_to, size_t store_to_ld, const data_t *diff_dst,
            const data_t *src, int ithr) {
        jit_1x1_conv_call_s p = {};
        rtus_driver_f32_t<avx512_mic>::call_params_t rp = {};

        p.output_stride = store_to_ld * sizeof(float);
        const int sp_step_def = jcp.nb_reduce_blocking * jcp.reduce_block;

        int oc_b_step = 0;
        for (int oc_b = 0; oc_b < nb_oc_blocking; oc_b += oc_b_step) {
            // !!!??? Why 12 and 18 ?
            oc_b_step = step(12, nb_oc_blocking - oc_b, 18);
            p.load_dim = oc_b_step * jcp.oc_block;

            int ic_b_step = 0;
            for (int ic_b = 0; ic_b < nb_ic_blocking; ic_b += ic_b_step) {
                // !!!??? Why 12 and 18 ?
                ic_b_step = step(12, nb_ic_blocking - ic_b, 18);
                p.bcast_dim = ic_b_step * jcp.ic_block;
                rp.icb = p.bcast_dim / jcp.ic_block;

                p.output_data = store_to + oc_b * store_to_ld
                    + ic_b * jcp.ic_block * jcp.oc_block;

                /* spatial reduction */
                int sp_step = 0;
                for (int sp = sp_start; sp < sp_end; sp += sp_step) {
                    sp_step = step(sp_step_def, sp_end - sp, 192);
                    p.reduce_dim = sp_step;
                    rp.os = p.reduce_dim;

                    p.reduce_pos_flag = sp == sp_start && first_image
                        ? jit_avx512_mic_1x1_conv_kernel_f32::REDUCE_FLAG_FIRST : 0;

                    p.load_data = diff_dst
                        + (oc_b * jcp.reduce_dim + sp) * jcp.oc_block;

                    if (conf_.rtus_.reduce_src_) {
                        const int oh = sp / jcp.ow;
                        const int ow = sp % jcp.ow;

                        const int ih = nstl::max(oh * stride_h - pad_t, 0);
                        const int iw = nstl::max(ow * stride_w - pad_l, 0);
                        rp.iw_start = iw;

                        rp.ws = scratch_ + ithr * ws_per_thread_
                            + (ic_b * jcp.is + sp) * jcp.ic_block;
                        rp.src = src
                            + ih * src_d.blocking_desc().strides[0][2]
                            + iw * src_d.blocking_desc().strides[0][3];

                        if (oc_b == 0)
                            rtus_driver_->ker_(&rp);

                        p.bcast_data = rp.ws;
                    } else
                        p.bcast_data = src
                            + (ic_b * jcp.reduce_dim + sp) * jcp.ic_block;

                    kernel_->jit_ker(&p);
                }
            }
        }
    };

    auto ker = [&](const int ithr, const int nthr) {
        auto rw = this->reducer_weights_;
        assert(nthr == rw->balancer_.nthr_);

        const int w_njobs = rw->balancer_.ithr_njobs(ithr);
        if (w_njobs == 0) return;

        /* setup: independent work (oc, ic) */
        const int w_job_start = rw->balancer_.ithr_job_off(ithr);
        int g{0}, load_i{0}, bcast_i{0};
        nd_iterator_init(w_job_start, g, jcp.ngroups, load_i, load_work,
                bcast_i, bcast_work);

        /* setup: reduction work (mb, sp) */
        int mb_sp_start{0}, mb_sp_end{0};
        balance211(mb_sp_work, rw->balancer_.nthr_per_group_,
                rw->balancer_.id_in_group(ithr), mb_sp_start, mb_sp_end);
        int img_start{0}, sp_start{0};
        nd_iterator_init(mb_sp_start, img_start, jcp.mb, sp_start, sp_dim);

        /* independent work */
        for (int iwork = 0; iwork < w_njobs; ++iwork) {
            const int oc_b = nb_oc_blocking * load_i;
            const int ic_b = nb_ic_blocking * bcast_i;

            const int _ic_b = g * nb_ic + ic_b;
            const int _oc_b = g * nb_oc + oc_b;

            data_t *store_to;
            size_t store_to_ld;

            if (rw->balancer_.nthr_per_group_ == 1 ||
                    (rw->balancer_.master(ithr) && rw->master_uses_dst_)) {
                const size_t off = conf_.with_groups()
                    ? diff_weights_d.blk_off(g, oc_b, ic_b)
                    : diff_weights_d.blk_off(oc_b, ic_b);
                store_to = &diff_weights[off];
                store_to_ld = jcp.ic * jcp.oc_block;
            } else {
                const size_t off = iwork * rw->balancer_.job_size_;
                store_to = &rw->get_local_ptr(ithr, nullptr)[off];
                store_to_ld = nb_ic_blocking * jcp.ic_block * jcp.oc_block;
            }

            /* reduction work */
            int img = img_start;
            int sp = sp_start;
            int sp_step = 0;
            for (int mb_sp = mb_sp_start; mb_sp < mb_sp_end; mb_sp += sp_step)
            {
                sp_step = nstl::min(sp_dim - sp, mb_sp_end - mb_sp);

                const bool first_image = img == img_start;
                oc_ic_sp_loop(sp, sp + sp_step, first_image, store_to,
                        store_to_ld, &diff_dst[diff_dst_d.blk_off(img, _oc_b)],
                        &src[src_d.blk_off(img, _ic_b)], ithr);

                sp = 0;
                img += 1;
            }

            nd_iterator_step(g, jcp.ngroups, load_i, load_work, bcast_i,
                             bcast_work);
        }
        rw->reduce(ithr, diff_weights);
    };

    auto ker_bias = [&](int ithr, int nthr) {
        auto rb = this->reducer_bias_;
        assert(nthr == rb->balancer_.nthr_);

        const int b_job_start = rb->balancer_.ithr_job_off(ithr);
        const int b_njobs = rb->balancer_.ithr_njobs(ithr);

        if (b_njobs == 0) return;

        /* reduction dimension */
        int img_start{0}, img_end{0};
        balance211(jcp.mb, rb->balancer_.nthr_per_group_,
                rb->balancer_.id_in_group(ithr), img_start, img_end);

        /* jobs */
        int g_start{0}, ocb_start{0};
        nd_iterator_init(b_job_start, g_start, jcp.ngroups, ocb_start, nb_oc);

        for (int img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start;
            for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
                const size_t _oc = g * nb_oc + ocb;

                const data_t *d_dst = &diff_dst[diff_dst_d.blk_off(img, _oc)];
                data_t *d_bias = &rb->get_local_ptr(ithr, diff_bias)[
                    b_job_loc * rb->balancer_.job_size_];

                if (img == img_start)
                    for (int o = 0; o < simd_w; ++o) d_bias[o] = 0.;
                for (int hw = 0; hw < jcp.oh * jcp.ow; ++hw) {
                    for (int o = 0; o < simd_w; ++o)
                        d_bias[o] += d_dst[o];
                    d_dst += simd_w;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, nb_oc);
            }
        }
        rb->reduce(ithr, diff_bias);
    };

#   pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        int nthr = omp_get_num_threads();
        ker(ithr, nthr);
        if (conf_.with_bias())
            ker_bias(ithr, nthr);
    }
}
}
}
}
