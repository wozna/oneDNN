/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;
using dim_t = dnnl::memory::dim;

void read_data(std::string file_name, std::vector<float> &buffor) {
    std::string prefix = "../../examples/inputs/";
    std::fstream myfile(prefix + file_name, std::ios_base::in);
    unsigned int i = 0;
    double a;

    while (myfile >> a && i < buffor.size())
        buffor[i++] = a;

    std::cout << "Finished reading \"" << file_name << "\" file. Size = " << i
              << "\n";
}

void gru_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);
    // left2right

    // Tensor dimensions.
    const memory::dim N = 100, // batch size
            T = 47, // time steps
            IC = 256, // input channels
            OC = 128, // output channels
            G = 3, // gates
            L = 1, // layers
            D = 1; // directions

    // Source (src), weights, bias, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {T, N, IC};
    memory::dims src_iter_dims = {L, D, N, OC};
    memory::dims weights_layer_dims = {L, D, IC, G, OC};
    memory::dims weights_iter_dims = {L, D, OC, G, OC};
    memory::dims bias_dims = {L, D, G, OC};
    memory::dims dst_dims = {T, N, OC};

    // Allocate buffers.
    std::vector<float> user_src_layer_data(product(src_dims));
    std::vector<float> user_src_iter_data(product(src_iter_dims), 0.0f);
    std::vector<float> user_weights_layer_data(product(weights_layer_dims));
    std::vector<float> user_weights_iter_data(product(weights_iter_dims));
    std::vector<float> user_dst_layer_data(product(dst_dims));
    std::vector<float> ref_dst_layer_data(product(dst_dims));
    std::vector<float> quant_dst_layer_data(product(dst_dims), 0.0f);
    std::vector<float> user_bias_data(product(bias_dims));
    std::vector<float> weights_scales(product(bias_dims));

    // Read date from files
    read_data("Input.txt", user_src_layer_data);
    read_data("WeightsX.txt", user_weights_layer_data);
    read_data("WeightsH.txt", user_weights_iter_data);
    read_data("WeightScale.txt", weights_scales);
    read_data("Bias.txt", user_bias_data);
    read_data("Output.txt", ref_dst_layer_data);

    // Create memory descriptors and memory objects for src, bias, and dst.
    auto user_src_layer_md = memory::desc(src_dims, dt::f32, tag::tnc);
    auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::ldgo);
    auto user_dst_layer_md = memory::desc(dst_dims, dt::f32, tag::tnc);

    auto user_src_layer_mem = memory(user_src_layer_md, engine);
    auto user_bias_mem = memory(user_bias_md, engine);
    auto user_dst_layer_mem = memory(user_dst_layer_md, engine);

    // Create memory objects for weights using user's memory layout. In this
    // example, LDIGO is assumed.
    auto user_weights_layer_mem
            = memory({weights_layer_dims, dt::f32, tag::ldigo}, engine);
    auto user_weights_iter_mem
            = memory({weights_iter_dims, dt::f32, tag::ldigo}, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(user_src_layer_data.data(), user_src_layer_mem);
    write_to_dnnl_memory(user_bias_data.data(), user_bias_mem);
    write_to_dnnl_memory(
            user_weights_layer_data.data(), user_weights_layer_mem);
    write_to_dnnl_memory(user_weights_iter_data.data(), user_weights_iter_mem);

    {
        auto gru_weights_layer_md
                = memory::desc(weights_layer_dims, dt::f32, tag::any);
        auto gru_weights_iter_md
                = memory::desc(weights_iter_dims, dt::f32, tag::any);
        auto src_iter_md = memory::desc(src_iter_dims, dt::f32, tag::any);

        // Create operation descriptor.
        auto gru_desc = gru_forward::desc(prop_kind::forward_inference,
                rnn_direction::unidirectional_left2right, user_src_layer_md,
                src_iter_md, gru_weights_layer_md, gru_weights_iter_md,
                user_bias_md, user_dst_layer_md, dnnl::memory::desc());

        auto gru_pd = gru_forward::primitive_desc(gru_desc, engine);

        auto gru_weights_layer_mem = user_weights_layer_mem;
        auto gru_weights_iter_mem = user_weights_iter_mem;
        auto bias_mem = user_bias_mem;

        if (gru_pd.weights_desc() != user_weights_layer_mem.get_desc()) {
            gru_weights_layer_mem = memory(gru_pd.weights_desc(), engine);
            reorder(user_weights_layer_mem, gru_weights_layer_mem)
                    .execute(engine_stream, user_weights_layer_mem,
                            gru_weights_layer_mem);
        }

        if (gru_pd.weights_iter_desc() != user_weights_iter_mem.get_desc()) {
            gru_weights_iter_mem = memory(gru_pd.weights_iter_desc(), engine);
            reorder(user_weights_iter_mem, gru_weights_iter_mem)
                    .execute(engine_stream, user_weights_iter_mem,
                            gru_weights_iter_mem);
        }

        if (gru_pd.bias_desc() != user_bias_mem.get_desc()) {
            bias_mem = memory(gru_pd.bias_desc(), engine);
            reorder(user_bias_mem, bias_mem)
                    .execute(engine_stream, user_bias_mem, bias_mem);
        }

        auto src_iter_mem = memory(
                gru_pd.src_iter_desc(), engine, user_src_iter_data.data());
        auto weights_iter_mem = memory(gru_pd.weights_iter_desc(), engine);
        auto workspace_mem = memory(gru_pd.workspace_desc(), engine);

        auto gru_prim = gru_forward(gru_pd);

        // Primitive arguments
        std::unordered_map<int, memory> gru_args;
        gru_args.insert({DNNL_ARG_SRC_LAYER, user_src_layer_mem});
        gru_args.insert({DNNL_ARG_SRC_ITER, src_iter_mem});
        gru_args.insert({DNNL_ARG_WEIGHTS_LAYER, gru_weights_layer_mem});
        gru_args.insert({DNNL_ARG_WEIGHTS_ITER, gru_weights_iter_mem});
        gru_args.insert({DNNL_ARG_BIAS, bias_mem});
        gru_args.insert({DNNL_ARG_DST_LAYER, user_dst_layer_mem});
        gru_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});

        gru_prim.execute(engine_stream, gru_args);
        engine_stream.wait();

        read_from_dnnl_memory(user_dst_layer_data.data(), user_dst_layer_mem);
    }

    ////////////////////////////// INT8

    // Quantization factors for f32 data
    const float data_shift = 128.f;
    const float data_scale = 105.859f;
    const int weights_scale_mask = 0
            + (1 << 3) // bit, indicating the unique scales for `g` dim in `ldigo`
            + (1 << 4); // bit, indicating the unique scales for `o` dim in `ldigo`

    primitive_attr attr;
    attr.set_rnn_data_qparams(data_scale, data_shift);
    attr.set_rnn_weights_qparams(weights_scale_mask, weights_scales);

    // Create memory descriptors for weights with format_tag::any. This enables
    // the GRU primitive to choose the optimized memory layout.
    auto quant_gru_weights_layer_md
            = memory::desc(weights_layer_dims, dt::s8, tag::any);
    auto quant_gru_weights_iter_md
            = memory::desc(weights_iter_dims, dt::s8, tag::any);
    auto quant_src_layer_md = memory::desc(src_dims, dt::u8, tag::any);
    auto quant_src_iter_md = memory::desc(src_iter_dims, dt::u8, tag::any);
    auto quant_bias_md = memory::desc(bias_dims, dt::f32, tag::ldgo);
    auto quant_dst_layer_md = memory::desc(dst_dims, dt::f32, tag::any);

    // Create operation descriptor.
    auto quant_gru_desc = gru_forward::desc(prop_kind::forward_inference,
            rnn_direction::unidirectional_left2right, quant_src_layer_md,
            quant_src_iter_md, quant_gru_weights_layer_md,
            quant_gru_weights_iter_md, quant_bias_md, quant_dst_layer_md,
            dnnl::memory::desc());

    auto quant_gru_pd
            = gru_forward::primitive_desc(quant_gru_desc, attr, engine);

    auto quant_src_layer_mem = user_src_layer_mem;
    auto quant_gru_weights_layer_mem = user_weights_layer_mem;
    auto quant_gru_weights_iter_mem = user_weights_iter_mem;
    auto quant_bias_mem = user_bias_mem;

    if (quant_gru_pd.src_desc() != user_src_layer_mem.get_desc()) {
        quant_src_layer_mem = memory(quant_gru_pd.src_desc(), engine);
        reorder(user_src_layer_mem, quant_src_layer_mem, attr)
                .execute(
                        engine_stream, user_src_layer_mem, quant_src_layer_mem);
    }

    if (quant_gru_pd.weights_desc() != user_weights_layer_mem.get_desc()) {
        quant_gru_weights_layer_mem
                = memory(quant_gru_pd.weights_desc(), engine);
        reorder(user_weights_layer_mem, quant_gru_weights_layer_mem, attr)
                .execute(engine_stream, user_weights_layer_mem,
                        quant_gru_weights_layer_mem);
    }

    if (quant_gru_pd.weights_iter_desc() != user_weights_iter_mem.get_desc()) {
        quant_gru_weights_iter_mem
                = memory(quant_gru_pd.weights_iter_desc(), engine);
        reorder(user_weights_iter_mem, quant_gru_weights_iter_mem, attr)
                .execute(engine_stream, user_weights_iter_mem,
                        quant_gru_weights_iter_mem);
    }

    if (quant_gru_pd.bias_desc() != user_bias_mem.get_desc()) {
        quant_bias_mem = memory(quant_gru_pd.bias_desc(), engine);
        reorder(user_bias_mem, quant_bias_mem)
                .execute(engine_stream, user_bias_mem, quant_bias_mem);
    }

    auto quant_src_iter_mem = memory(
            quant_gru_pd.src_iter_desc(), engine, user_src_iter_data.data());
    auto quant_weights_iter_mem
            = memory(quant_gru_pd.weights_iter_desc(), engine);
    auto quant_dst_layer_mem = memory(user_dst_layer_md, engine);
    auto quant_workspace_mem = memory(quant_gru_pd.workspace_desc(), engine);

    // Create the primitive.
    auto quant_gru_prim = gru_forward(quant_gru_pd);

    // Primitive arguments
    std::unordered_map<int, memory> quant_gru_args;
    quant_gru_args.insert({DNNL_ARG_SRC_LAYER, quant_src_layer_mem});
    quant_gru_args.insert({DNNL_ARG_SRC_ITER, quant_src_iter_mem});
    quant_gru_args.insert(
            {DNNL_ARG_WEIGHTS_LAYER, quant_gru_weights_layer_mem});
    quant_gru_args.insert({DNNL_ARG_WEIGHTS_ITER, quant_gru_weights_iter_mem});
    quant_gru_args.insert({DNNL_ARG_BIAS, quant_bias_mem});
    quant_gru_args.insert({DNNL_ARG_DST_LAYER, quant_dst_layer_mem});
    quant_gru_args.insert({DNNL_ARG_WORKSPACE, quant_workspace_mem});

    // Primitive execution: LSTM.
    quant_gru_prim.execute(engine_stream, quant_gru_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(quant_dst_layer_data.data(), quant_dst_layer_mem);

    double MSE = 0;
    for (size_t i = 0; i < quant_dst_layer_data.size(); i++)
        MSE += pow(quant_dst_layer_data[i] - user_dst_layer_data[i], 2.0);

    std::cout << "MSE=" << MSE << std::endl;

    //////////////////////////// INT8
}

int main(int argc, char **argv) {
    return handle_example_errors(gru_example, parse_engine_kind(argc, argv));
}
