/*
 * Copyright 2019 Xilinx, Inc.
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
 */

#ifndef _XF_RESIZE_CONFIG_
#define _XF_RESIZE_CONFIG_

#include "hls_stream.h"
#include "ap_int.h"
#include <vitis_common/common/xf_common.hpp>
#include <vitis_common/imgproc/xf_resize.hpp>

/* resize kernel configuration */

#define RO 0 // Resource Optimized (8-pixel implementation)
#define NO 1 // Normal Operation (1-pixel implementation)

// port widths
#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 128

// For Nearest Neighbor & Bilinear Interpolation, max down scale factor 2 for all 1-pixel modes, and for upscale in x
// direction
#define MAXDOWNSCALE 2

#define RGB 1
#define GRAY 0
/* Interpolation type*/
#define INTERPOLATION 1
// 0 - Nearest Neighbor Interpolation
// 1 - Bilinear Interpolation
// 2 - AREA Interpolation

/* Input image Dimensions */
#define WIDTH 640  // Maximum Input image width
#define HEIGHT 480 // Maximum Input image height

/* Output image Dimensions */
#define NEWWIDTH 1280  // Maximum output image width
#define NEWHEIGHT 960 // Maximum output image height

/* Interface types */
#if RO

#if RGB
#define NPC_T XF_NPPC4
#else
#define NPC_T XF_NPPC8
#endif

#else
#define NPC_T XF_NPPC1
#endif

#if RGB
#define TYPE XF_8UC3
#define CH_TYPE XF_RGB
#else
#define TYPE XF_8UC1
#define CH_TYPE XF_GRAY
#endif

#endif