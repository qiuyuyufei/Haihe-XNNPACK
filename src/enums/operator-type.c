// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/enums/operator-type.yaml
//   Generator: tools/generate-enum.py


#include <assert.h>
#include <stdint.h>

#include <xnnpack/operator-type.h>


static const uint16_t offset[155] = {
  0, 8, 22, 36, 50, 64, 78, 92, 119, 147, 175, 203, 230, 257, 289, 321, 339, 357, 382, 408, 424, 440, 455, 470, 492,
  515, 538, 561, 584, 607, 630, 653, 676, 694, 717, 740, 764, 782, 805, 829, 853, 877, 901, 936, 971, 995, 1019, 1043,
  1057, 1072, 1087, 1113, 1139, 1165, 1191, 1223, 1255, 1281, 1308, 1335, 1352, 1369, 1403, 1437, 1451, 1465, 1479,
  1495, 1511, 1537, 1563, 1595, 1627, 1664, 1701, 1738, 1775, 1801, 1833, 1859, 1893, 1927, 1961, 1995, 2029, 2063,
  2093, 2123, 2143, 2163, 2184, 2205, 2226, 2247, 2271, 2295, 2318, 2341, 2359, 2377, 2392, 2407, 2425, 2443, 2462,
  2481, 2500, 2519, 2536, 2553, 2569, 2585, 2613, 2641, 2669, 2697, 2724, 2751, 2768, 2785, 2826, 2867, 2885, 2903,
  2921, 2939, 2954, 2970, 2986, 3004, 3022, 3040, 3066, 3093, 3120, 3137, 3154, 3176, 3198, 3227, 3256, 3275, 3294,
  3313, 3332, 3347, 3362, 3377, 3392, 3411, 3431, 3451, 3471, 3492, 3513
};

static const char data[] =
  "Invalid\0"
  "Abs (NC, F16)\0"
  "Abs (NC, F32)\0"
  "Add (ND, F16)\0"
  "Add (ND, F32)\0"
  "Add (ND, QS8)\0"
  "Add (ND, QU8)\0"
  "ArgMax Pooling (NHWC, F32)\0"
  "Average Pooling (NHWC, F16)\0"
  "Average Pooling (NHWC, F32)\0"
  "Average Pooling (NHWC, QU8)\0"
  "Bankers Rounding (NC, F16)\0"
  "Bankers Rounding (NC, F32)\0"
  "Batch Matrix Multiply (NC, F16)\0"
  "Batch Matrix Multiply (NC, F32)\0"
  "Ceiling (NC, F16)\0"
  "Ceiling (NC, F32)\0"
  "Channel Shuffle (NC, X8)\0"
  "Channel Shuffle (NC, X32)\0"
  "Clamp (NC, F16)\0"
  "Clamp (NC, F32)\0"
  "Clamp (NC, S8)\0"
  "Clamp (NC, U8)\0"
  "Constant Pad (ND, X8)\0"
  "Constant Pad (ND, X16)\0"
  "Constant Pad (ND, X32)\0"
  "Convert (NC, F16, F32)\0"
  "Convert (NC, F16, QD8)\0"
  "Convert (NC, F32, F16)\0"
  "Convert (NC, F32, QD8)\0"
  "Convert (NC, F32, QS8)\0"
  "Convert (NC, F32, QU8)\0"
  "Convert (NC, QS8)\0"
  "Convert (NC, QS8, F16)\0"
  "Convert (NC, QS8, F32)\0"
  "Convert (NC, QS16, QS8)\0"
  "Convert (NC, QU8)\0"
  "Convert (NC, QU8, F32)\0"
  "Convolution (NCHW, F16)\0"
  "Convolution (NCHW, F32)\0"
  "Convolution (NHWC, F16)\0"
  "Convolution (NHWC, F32)\0"
  "Convolution (NHWC, QD8, F16, QC8W)\0"
  "Convolution (NHWC, QD8, F32, QC8W)\0"
  "Convolution (NHWC, QC8)\0"
  "Convolution (NHWC, QS8)\0"
  "Convolution (NHWC, QU8)\0"
  "Copy (NC, X8)\0"
  "Copy (NC, X16)\0"
  "Copy (NC, X32)\0"
  "Deconvolution (NHWC, F16)\0"
  "Deconvolution (NHWC, F32)\0"
  "Deconvolution (NHWC, QS8)\0"
  "Deconvolution (NHWC, QU8)\0"
  "Depth To Space (NCHW2NHWC, X16)\0"
  "Depth To Space (NCHW2NHWC, X32)\0"
  "Depth To Space (NHWC, X8)\0"
  "Depth To Space (NHWC, X16)\0"
  "Depth To Space (NHWC, X32)\0"
  "Divide (ND, F16)\0"
  "Divide (ND, F32)\0"
  "Dynamic Fully Connected (NC, F16)\0"
  "Dynamic Fully Connected (NC, F32)\0"
  "ELU (NC, F16)\0"
  "ELU (NC, F32)\0"
  "ELU (NC, QS8)\0"
  "Floor (NC, F16)\0"
  "Floor (NC, F32)\0"
  "Fully Connected (NC, F16)\0"
  "Fully Connected (NC, F32)\0"
  "Fully Connected (NC, F32, QC4W)\0"
  "Fully Connected (NC, F32, QC8W)\0"
  "Fully Connected (NC, QD8, F16, QC8W)\0"
  "Fully Connected (NC, QD8, F16, QC4W)\0"
  "Fully Connected (NC, QD8, F32, QC4W)\0"
  "Fully Connected (NC, QD8, F32, QC8W)\0"
  "Fully Connected (NC, QS8)\0"
  "Fully Connected (NC, QS8, QC8W)\0"
  "Fully Connected (NC, QU8)\0"
  "Global Average Pooling (NCW, F16)\0"
  "Global Average Pooling (NCW, F32)\0"
  "Global Average Pooling (NWC, F16)\0"
  "Global Average Pooling (NWC, F32)\0"
  "Global Average Pooling (NWC, QS8)\0"
  "Global Average Pooling (NWC, QU8)\0"
  "Global Sum Pooling (NWC, F16)\0"
  "Global Sum Pooling (NWC, F32)\0"
  "HardSwish (NC, F16)\0"
  "HardSwish (NC, F32)\0"
  "Leaky ReLU (NC, F16)\0"
  "Leaky ReLU (NC, F32)\0"
  "Leaky ReLU (NC, QS8)\0"
  "Leaky ReLU (NC, QU8)\0"
  "Max Pooling (NHWC, F16)\0"
  "Max Pooling (NHWC, F32)\0"
  "Max Pooling (NHWC, S8)\0"
  "Max Pooling (NHWC, U8)\0"
  "Maximum (ND, F16)\0"
  "Maximum (ND, F32)\0"
  "Mean (ND, F16)\0"
  "Mean (ND, F32)\0"
  "Minimum (ND, F16)\0"
  "Minimum (ND, F32)\0"
  "Multiply (ND, F16)\0"
  "Multiply (ND, F32)\0"
  "Multiply (ND, QS8)\0"
  "Multiply (ND, QU8)\0"
  "Negate (NC, F16)\0"
  "Negate (NC, F32)\0"
  "PReLU (NC, F16)\0"
  "PReLU (NC, F32)\0"
  "Resize Bilinear (NCHW, F16)\0"
  "Resize Bilinear (NCHW, F32)\0"
  "Resize Bilinear (NHWC, F16)\0"
  "Resize Bilinear (NHWC, F32)\0"
  "Resize Bilinear (NHWC, S8)\0"
  "Resize Bilinear (NHWC, U8)\0"
  "RoPE (NTHC, F16)\0"
  "RoPE (NTHC, F32)\0"
  "Scaled Dot-Product Attention (NHTC, F16)\0"
  "Scaled Dot-Product Attention (NHTC, F32)\0"
  "Sigmoid (NC, F16)\0"
  "Sigmoid (NC, F32)\0"
  "Sigmoid (NC, QS8)\0"
  "Sigmoid (NC, QU8)\0"
  "Slice (ND, X8)\0"
  "Slice (ND, X16)\0"
  "Slice (ND, X32)\0"
  "Softmax (NC, F16)\0"
  "Softmax (NC, F32)\0"
  "Softmax (NC, QU8)\0"
  "Space To Depth (NHWC, X8)\0"
  "Space To Depth (NHWC, X16)\0"
  "Space To Depth (NHWC, X32)\0"
  "Square (NC, F16)\0"
  "Square (NC, F32)\0"
  "Square Root (NC, F16)\0"
  "Square Root (NC, F32)\0"
  "Squared Difference (NC, F16)\0"
  "Squared Difference (NC, F32)\0"
  "Subtract (ND, F16)\0"
  "Subtract (ND, F32)\0"
  "Subtract (ND, QS8)\0"
  "Subtract (ND, QU8)\0"
  "Tanh (NC, F16)\0"
  "Tanh (NC, F32)\0"
  "Tanh (NC, QS8)\0"
  "Tanh (NC, QU8)\0"
  "Transpose (ND, X8)\0"
  "Transpose (ND, X16)\0"
  "Transpose (ND, X32)\0"
  "Transpose (ND, X64)\0"
  "Truncation (NC, F16)\0"
  "Truncation (NC, F32)\0"
  "Unpooling (NHWC, X32)";

const char* xnn_operator_type_to_string(enum xnn_operator_type operator_type) {
  assert(operator_type >= xnn_operator_type_invalid);
  assert(operator_type <= xnn_operator_type_unpooling_nhwc_x32);
  return &data[offset[operator_type]];
}
