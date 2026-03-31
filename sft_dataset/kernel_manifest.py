#!/usr/bin/env python3
"""Build the kernel selection manifest for SFT dataset generation."""

import json, os

BASE = "/workdir/agentic-workflow-for-optimising-inference-kernels/generated_kernels"

KERNELS = [
    # Group 1: SIMT GEMM (baseline, no tensor cores)
    {"group": "simt_gemm", "concept": "Scalar thread-level GEMM without tensor cores. Baseline compute.",
     "file": f"{BASE}/gemm/50/sgemm/cutlass_simt_sgemm_128x128_8x2_nn_align1.cu"},
    {"group": "simt_gemm", "concept": "SIMT GEMM with transposed-A layout",
     "file": f"{BASE}/gemm/50/sgemm/cutlass_simt_sgemm_64x128_8x2_tn_align1.cu"},
    {"group": "simt_gemm", "concept": "SIMT double-precision GEMM",
     "file": f"{BASE}/gemm/50/dgemm/cutlass_simt_dgemm_64x64_8x2_nn_align1.cu"},
    {"group": "simt_gemm", "concept": "SIMT complex single-precision GEMM",
     "file": f"{BASE}/gemm/50/cgemm/cutlass_simt_cgemm_64x64_8x2_nn_align1.cu"},

    # Group 2: TensorOp GEMM across architectures
    {"group": "tensorop_gemm_evolution", "concept": "Volta tensor core GEMM (MMA 8x8x4, first gen TC)",
     "file": f"{BASE}/gemm/70/h884gemm/cutlass_tensorop_h884gemm_128x256_32x2_tn_align8.cu"},
    {"group": "tensorop_gemm_evolution", "concept": "Turing tensor core GEMM (MMA 16x8x8, 2nd gen)",
     "file": f"{BASE}/gemm/75/h1688gemm/cutlass_tensorop_h1688gemm_256x128_32x2_tn_align8.cu"},
    {"group": "tensorop_gemm_evolution", "concept": "Ampere tensor core GEMM (MMA 16x8x16, 3rd gen, 3 stages)",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x3_nn_align8.cu"},
    {"group": "tensorop_gemm_evolution", "concept": "Ampere tensor core GEMM large tile (256x128, 3 stages)",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_256x128_64x3_tn_align8.cu"},

    # Group 3: Conv2d Implicit GEMM directions
    {"group": "conv2d_directions", "concept": "Conv2d forward pass (fprop) as implicit GEMM",
     "file": f"{BASE}/conv2d/80/h16816fprop_optimized/cutlass_tensorop_h16816fprop_optimized_128x128_64x3_nhwc_align8.cu"},
    {"group": "conv2d_directions", "concept": "Conv2d data gradient (dgrad) as implicit GEMM",
     "file": f"{BASE}/conv2d/80/h16816dgrad_optimized/cutlass_tensorop_h16816dgrad_optimized_128x128_64x3_nhwc_align8.cu"},
    {"group": "conv2d_directions", "concept": "Conv2d weight gradient (wgrad) as implicit GEMM",
     "file": f"{BASE}/conv2d/80/h16816wgrad_optimized/cutlass_tensorop_h16816wgrad_optimized_128x128_64x3_nhwc_align8.cu"},

    # Group 4: Conv3d
    {"group": "conv3d", "concept": "Conv3d forward pass with TensorNDHWC layout",
     "file": f"{BASE}/conv3d/80/h16816fprop3d_analytic/cutlass_tensorop_h16816fprop3d_analytic_128x128_64x3.cu"},
    {"group": "conv3d", "concept": "Conv3d data gradient (3D)",
     "file": f"{BASE}/conv3d/80/h16816dgrad3d_analytic/cutlass_tensorop_h16816dgrad3d_analytic_128x128_64x3.cu"},
    {"group": "conv3d", "concept": "Conv3d weight gradient (3D)",
     "file": f"{BASE}/conv3d/80/h16816wgrad3d_analytic/cutlass_tensorop_h16816wgrad3d_analytic_128x128_64x3.cu"},

    # Group 5: Full precision
    {"group": "full_precision", "concept": "Single-precision SIMT GEMM on Ampere (5-stage pipeline)",
     "file": f"{BASE}/gemm/80/sgemm/cutlass_simt_sgemm_128x256_8x5_nn_align1.cu"},
    {"group": "full_precision", "concept": "Double-precision SIMT GEMM on Ampere",
     "file": f"{BASE}/gemm/80/dgemm/cutlass_simt_dgemm_64x64_8x5_nn_align1.cu"},
    {"group": "full_precision", "concept": "Double-precision tensor core GEMM (d884, Ampere only)",
     "file": f"{BASE}/gemm/80/d884gemm/cutlass_tensorop_d884gemm_64x64_16x4_tn_align1.cu"},

    # Group 6: Half precision native (f16 in, f16 out)
    {"group": "half_precision_native", "concept": "Native f16 GEMM with 4-stage pipeline (128x128)",
     "file": f"{BASE}/gemm/80/h16816gemm/cutlass_tensorop_h16816gemm_128x128_32x4_nn_align8.cu"},
    {"group": "half_precision_native", "concept": "Native f16 GEMM wide tile (256x128, 3 stages)",
     "file": f"{BASE}/gemm/80/h16816gemm/cutlass_tensorop_h16816gemm_256x128_32x3_tn_align8.cu"},
    {"group": "half_precision_native", "concept": "Native f16 GEMM small tile deep pipeline (64x64, 5 stages)",
     "file": f"{BASE}/gemm/80/h16816gemm/cutlass_tensorop_h16816gemm_64x64_64x5_nn_align8.cu"},

    # Group 7: Mixed precision (f16 in, f32 accum)
    {"group": "mixed_precision_f16_f32", "concept": "Mixed precision f16->f32 baseline (128x128, 3 stages, align8)",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x3_nn_align8.cu"},
    {"group": "mixed_precision_f16_f32", "concept": "Mixed precision f16->f32 low alignment (256x128, align2)",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_256x128_32x3_tn_align2.cu"},
    {"group": "mixed_precision_f16_f32", "concept": "Mixed precision f16->f32 tall-narrow tile (64x256, 4 stages)",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_64x256_32x4_nn_align8.cu"},

    # Group 8: BFloat16
    {"group": "bfloat16", "concept": "BF16 input and output with BF16 accumulation (128x128, 3 stages)",
     "file": f"{BASE}/gemm/80/bf16_s16816gemm_bf16/cutlass_tensorop_bf16_s16816gemm_bf16_128x128_32x3_nn_align8.cu"},
    {"group": "bfloat16", "concept": "BF16 large tile (256x128, K=64, 3 stages)",
     "file": f"{BASE}/gemm/80/bf16_s16816gemm_bf16/cutlass_tensorop_bf16_s16816gemm_bf16_256x128_64x3_tn_align8.cu"},
    {"group": "bfloat16", "concept": "BF16 input with f32 accumulation (128x128, 5 stages)",
     "file": f"{BASE}/gemm/80/s16816gemm_bf16/cutlass_tensorop_s16816gemm_bf16_128x128_32x5_nn_align8.cu"},

    # Group 9: TF32
    {"group": "tf32", "concept": "TF32 GEMM (f32 API, 16x8x8 MMA, 3 stages)",
     "file": f"{BASE}/gemm/80/s1688gemm_tf32/cutlass_tensorop_s1688gemm_tf32_128x128_16x3_nn_align4.cu"},
    {"group": "tf32", "concept": "TF32 with f32 input (auto-converted, 4 stages)",
     "file": f"{BASE}/gemm/80/tf32_s1688gemm_tf32/cutlass_tensorop_tf32_s1688gemm_tf32_128x128_16x4_tn_align4.cu"},
    {"group": "tf32", "concept": "TF32 output accumulation (128x128, 3 stages)",
     "file": f"{BASE}/gemm/80/s1688tf32gemm/cutlass_tensorop_s1688tf32gemm_128x128_16x3_nn_align4.cu"},

    # Group 10: Integer quantized
    {"group": "integer_quantized", "concept": "INT8 GEMM with 16x8x32 MMA (Ampere)",
     "file": f"{BASE}/gemm/80/i16832gemm_s8/cutlass_tensorop_i16832gemm_s8_128x32_64x6_tn_align16.cu"},
    {"group": "integer_quantized", "concept": "INT4 GEMM with 16x8x64 MMA (extreme quantization)",
     "file": f"{BASE}/gemm/80/i16864gemm_s4/cutlass_tensorop_i16864gemm_s4_128x128_256x3_tn_align32.cu"},
    {"group": "integer_quantized", "concept": "Binary (1-bit) XOR GEMM with 16x8x256 MMA",
     "file": f"{BASE}/gemm/80/i168256xorgemm_b1/cutlass_tensorop_i168256xorgemm_b1_256x128_1024x3_tn_align128.cu"},
    {"group": "integer_quantized", "concept": "UINT8 GEMM with u8 output and clamping",
     "file": f"{BASE}/gemm/80/u8_i16832gemm_u8/cutlass_tensorop_u8_i16832gemm_u8_32x256_128x4_tn_align16.cu"},

    # Group 14: Tile shape selection (same op, different tiles)
    {"group": "tile_shape_selection", "concept": "Small tile (64x64) with deep 10-stage pipeline",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_64x64_32x10_nn_align8.cu"},
    {"group": "tile_shape_selection", "concept": "Medium tile (128x128) balanced config",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x3_nn_align8.cu"},
    {"group": "tile_shape_selection", "concept": "Large wide tile (256x128) for large GEMM",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_256x128_32x3_nn_align8.cu"},
    {"group": "tile_shape_selection", "concept": "Tall narrow tile (64x256) for N-dominant workloads",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_64x256_64x3_nn_align8.cu"},

    # Group 15: Pipeline staging
    {"group": "pipeline_staging", "concept": "3-stage pipeline (48KB smem, fits default carveout)",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x3_nn_align8.cu"},
    {"group": "pipeline_staging", "concept": "4-stage pipeline (64KB smem, needs extended carveout)",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x4_nn_align8.cu"},
    {"group": "pipeline_staging", "concept": "5-stage pipeline (80KB smem, deep latency hiding)",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x5_nn_align8.cu"},
    {"group": "pipeline_staging", "concept": "4-stage with large K-tile (128KB smem, max reuse)",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_64x4_nn_align8.cu"},

    # Group 16: Alignment
    {"group": "alignment", "concept": "Low alignment (align2) for unaligned data",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x3_nn_align2.cu"},
    {"group": "alignment", "concept": "Medium alignment (align4)",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x3_nn_align4.cu"},
    {"group": "alignment", "concept": "Full alignment (align8) for optimal vectorized loads",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x3_nn_align8.cu"},

    # Group 17: Layout
    {"group": "layout", "concept": "NN layout (both row-major) - A row, B row",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x3_nn_align8.cu"},
    {"group": "layout", "concept": "NT layout (A row, B col) - common for A*B^T",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x3_nt_align8.cu"},
    {"group": "layout", "concept": "TN layout (A col, B row) - common for A^T*B",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8.cu"},
    {"group": "layout", "concept": "TT layout (both col-major) - A col, B col",
     "file": f"{BASE}/gemm/80/s16816gemm_f16/cutlass_tensorop_s16816gemm_f16_128x128_32x3_tt_align8.cu"},

    # Group 18: Analytic vs Optimized iterators
    {"group": "conv_iterator_strategy", "concept": "Analytic iterator (compute address on-the-fly, save smem)",
     "file": f"{BASE}/conv2d/80/h16816fprop_analytic/cutlass_tensorop_h16816fprop_analytic_128x128_64x3_nhwc_align8.cu"},
    {"group": "conv_iterator_strategy", "concept": "Optimized iterator (precomputed offsets, faster access)",
     "file": f"{BASE}/conv2d/80/h16816fprop_optimized/cutlass_tensorop_h16816fprop_optimized_128x128_64x3_nhwc_align8.cu"},
    {"group": "conv_iterator_strategy", "concept": "Analytic iterator for mixed-precision conv (f16 in, f32 accum)",
     "file": f"{BASE}/conv2d/80/s16816fprop_analytic_f16/cutlass_tensorop_s16816fprop_analytic_f16_128x128_32x3_nhwc_align8.cu"},
    {"group": "conv_iterator_strategy", "concept": "Optimized iterator for mixed-precision conv (f16 in, f32 accum)",
     "file": f"{BASE}/conv2d/80/s16816fprop_optimized_f16/cutlass_tensorop_s16816fprop_optimized_f16_128x128_32x3_nhwc_align8.cu"},

    # Group 19: Few/fixed channels
    {"group": "conv_channel_specialization", "concept": "Fixed channels for first conv layer (C<32, avoids padding waste)",
     "file": f"{BASE}/conv2d/80/h16816fprop_fixed_channels/cutlass_tensorop_h16816fprop_fixed_channels_128x128_64x3_nhwc_align8.cu"},
    {"group": "conv_channel_specialization", "concept": "Few channels INT8 fprop (SM75, packs multiple pixels)",
     "file": f"{BASE}/conv2d/75/s8_i8816fprop_few_channels_s8/cutlass_tensorop_s8_i8816fprop_few_channels_s8_128x128_32x2_nhwc_align16.cu"},
    {"group": "conv_channel_specialization", "concept": "Fixed channels INT8 on Ampere (5-stage deep pipeline)",
     "file": f"{BASE}/conv2d/80/s8_i16832fprop_fixed_channels_s8/cutlass_tensorop_s8_i16832fprop_fixed_channels_s8_128x128_64x5_nhwc_align16.cu"},
    {"group": "conv_channel_specialization", "concept": "Few channels INT8 on Ampere (5-stage deep pipeline)",
     "file": f"{BASE}/conv2d/80/s8_i16832fprop_few_channels_s8/cutlass_tensorop_s8_i16832fprop_few_channels_s8_128x128_64x5_nhwc_align16.cu"},

    # Group 20: Depthwise conv
    {"group": "depthwise_conv", "concept": "Depthwise 5x5 conv, stride 2, 64-channel output tile",
     "file": f"{BASE}/conv2d/60/hfprop_fixed_stride_dilation/cutlass_simt_hfprop_fixed_stride_dilation_64x16x25_1x8x8x16_3_filter5x5_stride2x2_dilation1x1_nhwc_depthwise_align8.cu"},
    {"group": "depthwise_conv", "concept": "Depthwise 5x5 conv with dilation 2 (dilated convolution)",
     "file": f"{BASE}/conv2d/60/hfprop_fixed_stride_dilation/cutlass_simt_hfprop_fixed_stride_dilation_16x16x25_1x4x4x16_4_filter5x5_stride2x2_dilation2x2_nhwc_depthwise_align8.cu"},
    {"group": "depthwise_conv", "concept": "Depthwise 3x3 conv with dilation 2, 64-channel",
     "file": f"{BASE}/conv2d/60/hfprop_fixed_stride_dilation/cutlass_simt_hfprop_fixed_stride_dilation_16x64x9_1x4x4x64_4_filter3x3_stride1x1_dilation2x2_nhwc_depthwise_align8.cu"},

    # Group 22: TRMM
    {"group": "trmm", "concept": "Triangular matrix multiply (f32, left-side, lower, non-unit diagonal)",
     "file": f"{BASE}/trmm/80/s1688trmm/cutlass_tensorop_s1688trmm_128x128_16x5_nn_ls_l_nu_align4.cu"},
    {"group": "trmm", "concept": "Complex double TRMM (right-side, upper, non-unit)",
     "file": f"{BASE}/trmm/80/z884trmm/cutlass_tensorop_z884trmm_32x64_8x4_nn_rs_u_un_align1.cu"},
    {"group": "trmm", "concept": "Complex float TRMM (conj-transpose, right-side, lower)",
     "file": f"{BASE}/trmm/80/c1688trmm/cutlass_tensorop_c1688trmm_64x128_16x4_cn_rs_l_un_align1.cu"},

    # Group 23: SYRK/HERK
    {"group": "rank_k", "concept": "Symmetric rank-k update (SYRK, f32, lower fill)",
     "file": f"{BASE}/rank_k/80/s1688syrk/cutlass_tensorop_s1688syrk_128x128_16x5_n_l_align4.cu"},
    {"group": "rank_k", "concept": "Hermitian rank-k update (HERK, complex float)",
     "file": f"{BASE}/rank_k/80/c1688herk/cutlass_tensorop_c1688herk_128x64_16x4_h_l_align1.cu"},
    {"group": "rank_k", "concept": "Hermitian rank-k update (complex double, z884 MMA)",
     "file": f"{BASE}/rank_k/80/z884herk/cutlass_tensorop_z884herk_64x64_8x3_h_l_align1.cu"},

    # Group 24: SYMM/HEMM
    {"group": "symm_hemm", "concept": "Symmetric matrix multiply (f32, left-side, lower fill)",
     "file": f"{BASE}/symm/80/s1688symm/cutlass_tensorop_s1688symm_128x128_32x4_n_ls_l_align4.cu"},
    {"group": "symm_hemm", "concept": "Hermitian matrix multiply (complex float, left-side, upper)",
     "file": f"{BASE}/symm/80/c1688hemm/cutlass_tensorop_c1688hemm_64x64_16x4_n_ls_u_align1.cu"},
    {"group": "symm_hemm", "concept": "Hermitian matrix multiply (complex double, z884 MMA)",
     "file": f"{BASE}/symm/80/z884hemm/cutlass_tensorop_z884hemm_64x128_8x3_n_ls_u_align1.cu"},

    # Group 25: Planar complex
    {"group": "planar_complex", "concept": "Planar complex GEMM (separate real/imag planes, f16)",
     "file": f"{BASE}/gemm/80/h16816gemm_planar_complex/cutlass_tensorop_h16816gemm_planar_complex_64x64_32x4_ct_align8.cu"},
    {"group": "planar_complex", "concept": "Planar complex with f32 accum (mixed precision complex)",
     "file": f"{BASE}/gemm/80/s16816gemm_planar_complex_f16/cutlass_tensorop_s16816gemm_planar_complex_f16_64x128_32x3_cc_align8.cu"},
    {"group": "planar_complex", "concept": "Planar complex BF16 GEMM",
     "file": f"{BASE}/gemm/80/bf16_s16816gemm_planar_complex_bf16/cutlass_tensorop_bf16_s16816gemm_planar_complex_bf16_128x64_32x3_ct_align8.cu"},

    # Group 26: Grouped GEMM
    {"group": "grouped_gemm", "concept": "Grouped GEMM f16->f32 (device-scheduled, variable sizes)",
     "file": f"{BASE}/gemm/80/s16816gemm_grouped_f16/cutlass_tensorop_s16816gemm_grouped_f16_128x128_64x4_nt_align2_scheduleDevice.cu"},
    {"group": "grouped_gemm", "concept": "Grouped GEMM bf16 (device-scheduled, large tile)",
     "file": f"{BASE}/gemm/80/s16816gemm_grouped_bf16/cutlass_tensorop_s16816gemm_grouped_bf16_128x256_64x3_tt_align4_scheduleDevice.cu"},
    {"group": "grouped_gemm", "concept": "Grouped GEMM native f16 (device-scheduled)",
     "file": f"{BASE}/gemm/80/h16816gemm_grouped/cutlass_tensorop_h16816gemm_grouped_64x128_32x6_nt_align4_scheduleDevice.cu"},
]

# Validate all files exist
missing = [k for k in KERNELS if not os.path.exists(k["file"])]
if missing:
    for m in missing:
        print(f"MISSING: {m['file']}")
else:
    print(f"All {len(KERNELS)} kernel files found.")

# Read all kernel contents
for k in KERNELS:
    with open(k["file"]) as f:
        k["code"] = f.read()

# Write manifest
with open("/workdir/agentic-workflow-for-optimising-inference-kernels/sft_dataset/kernel_manifest.json", "w") as f:
    json.dump(KERNELS, f, indent=2)

# Stats
from collections import Counter
groups = Counter(k["group"] for k in KERNELS)
print(f"\nTotal kernels: {len(KERNELS)}")
print(f"Groups: {len(groups)}")
for g, c in sorted(groups.items()):
    print(f"  {g}: {c}")
