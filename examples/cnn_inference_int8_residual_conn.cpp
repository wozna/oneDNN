/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

/// @example cnn_inference_int8.cpp
/// @copybrief cnn_inference_int8_cpp
/// > Annotated version: @ref cnn_inference_int8_cpp

/// @page cnn_inference_int8_cpp CNN int8 inference example
/// This C++ API example demonstrates how to run AlexNet's conv3 and relu3
/// with int8 data type.
///
/// > Example code: @ref cnn_inference_int8.cpp

#include <stdexcept>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

void simple_net_int8(engine::kind engine_kind) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  auto eng = engine(engine_kind, 0);
  stream s(eng);

  const int batch = 1;

  memory::dims conv_src_tz = {batch, 1, 5, 5};
  memory::dims conv_weights_tz = {2, 1, 3, 3};
  memory::dims input_residual_size_tz = {1, 2, 3, 3};
  memory::dims conv_dst_tz = {batch, 2, 3, 3};
  memory::dims conv_strides = {1, 1};
  memory::dims conv_padding = {0, 0};

  const std::vector<float> src_scales = {0.95f};
  const std::vector<float> weight_scales = {12.0f};
  const std::vector<float> residual_scales = {0.5f};
  const std::vector<float> dst_scales = {0.5f};

  const int src_mask = 0;
  const int weight_mask = 0;
  const int conv_mask = 0;

  std::vector<float> user_src({-5.,  -4.6, -4.2, -3.8, -3.4, -3,   -2.6,
                               -2.2, -1.8, -1.4, -1.,  -0.6, -0.2, 0.2,
                               0.6,  1,    1.4,  1.8,  2.2,  2.6,  3,
                               3.4,  3.8,  4.2,  4.6});
  std::vector<float> user_dst(
      {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});

  std::vector<float> conv_weights({0.69, 0.55, 0.98, 0.39, 0.43, 0.73, 0.69,
                                   0.55, 0.98, 0.39, 0.43, 0.73, 0.69, 0.55,
                                   0.98, 0.39, 0.43, 0.73});

  auto user_src_memory = memory({{conv_src_tz}, dt::f32, tag::nchw}, eng);
  write_to_dnnl_memory(user_src.data(), user_src_memory);
  auto user_weights_memory =
      memory({{conv_weights_tz}, dt::f32, tag::oihw}, eng);
  write_to_dnnl_memory(conv_weights.data(), user_weights_memory);

  auto conv_src_md = memory::desc({conv_src_tz}, dt::s8, tag::any);
  auto conv_weights_md = memory::desc({conv_weights_tz}, dt::s8, tag::any);
  auto conv_dst_md = memory::desc({conv_dst_tz}, dt::u8, tag::any);

  auto conv_desc = convolution_forward::desc(
      prop_kind::forward, algorithm::convolution_direct, conv_src_md,
      conv_weights_md, conv_dst_md, conv_strides, conv_padding, conv_padding);

  primitive_attr conv_attr;
  float output_scale = (dst_scales[0] / (src_scales[0] * weight_scales[0]));
  float sum_scale = (dst_scales[0] / residual_scales[0]);
  conv_attr.set_output_scales(conv_mask, {output_scale});

  const float ops_scale = 1.f;
  const float ops_alpha = 0.f; // relu negative slope
  const float ops_beta = 0.f;
  post_ops ops;
  // dst[:] = Relu ( sum_scale * dst[:] + conv(src[:], weights[:]))
  ops.append_sum(sum_scale);
  ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
  conv_attr.set_post_ops(ops);

  try {
    convolution_forward::primitive_desc(conv_desc, conv_attr, eng);
  } catch (error &e) {
    if (e.status == dnnl_unimplemented)
      throw example_allows_unimplemented{
          "No int8 convolution implementation is available for this "
          "platform.\n"
          "Please refer to the developer guide for details."};

    // on any other error just re-throw
    throw;
  }

  auto conv_prim_desc =
      convolution_forward::primitive_desc(conv_desc, conv_attr, eng);

  auto conv_src_memory = memory(conv_prim_desc.src_desc(), eng);
  primitive_attr src_attr;
  src_attr.set_output_scales(src_mask, src_scales);
  auto src_reorder_pd =
      reorder::primitive_desc(eng, user_src_memory.get_desc(), eng,
                              conv_src_memory.get_desc(), src_attr);
  auto src_reorder = reorder(src_reorder_pd);
  src_reorder.execute(s, user_src_memory, conv_src_memory);

  auto conv_weights_memory = memory(conv_prim_desc.weights_desc(), eng);
  primitive_attr weight_attr;
  weight_attr.set_output_scales(weight_mask, weight_scales);
  auto weight_reorder_pd =
      reorder::primitive_desc(eng, user_weights_memory.get_desc(), eng,
                              conv_weights_memory.get_desc(), weight_attr);
  auto weight_reorder = reorder(weight_reorder_pd);
  weight_reorder.execute(s, user_weights_memory, conv_weights_memory);

  auto conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);
  auto conv_bias_memory = memory(conv_prim_desc.bias_desc(), eng);

  auto conv = convolution_forward(conv_prim_desc);
  conv.execute(s, {{DNNL_ARG_SRC, conv_src_memory},
                   {DNNL_ARG_WEIGHTS, conv_weights_memory},
                   {DNNL_ARG_BIAS, conv_bias_memory},
                   {DNNL_ARG_DST, conv_dst_memory}});

  auto user_dst_memory = memory({{conv_dst_tz}, dt::f32, tag::nchw}, eng);
  write_to_dnnl_memory(user_dst.data(), user_dst_memory);
  primitive_attr dst_attr;
  // dst_attr.set_output_scales(dst_mask, dst_scales);
  auto dst_reorder_pd =
      reorder::primitive_desc(eng, conv_dst_memory.get_desc(), eng,
                              user_dst_memory.get_desc(), dst_attr);
  auto dst_reorder = reorder(dst_reorder_pd);
  dst_reorder.execute(s, conv_dst_memory, user_dst_memory);

  s.wait();

  // dequatize output
  read_from_dnnl_memory(user_dst.data(), user_dst_memory);
  for (uint i = 0; i < user_dst.size(); i++) {
    std::cout << user_dst[i] * dst_scales[0] << " ";
  }

  // result for tag v2.3.2 = 0 0 0 0 0 0.5 2 2.5 4 0 0 0 0 0 0.5 2 2.5 3
  // result for tag v2.4 = 13 13.5 14.5 16 16.5 17.5 19 19.5 20 13.5 14 14.5 16
  // 16.5 17.5 19 19 19.5
}

int main(int argc, char **argv) {
  return handle_example_errors(simple_net_int8, parse_engine_kind(argc, argv));
}
