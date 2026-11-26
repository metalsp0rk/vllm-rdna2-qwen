// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-side entry points for the fork's RDNA2 decode kernels: the MoE int4
// decode family (v1/v2/v3) and the dense skinny GEMVs (fp16, int8). v0.34.0
// split the int4 skinny GEMM into the api + five N_in units sharing
// skinny_gemms_int4.cuh; the fork's host helpers date from the single-TU era
// and used to be defined in that header, which made every unit emit them
// (duplicate symbols) or, when inline, left torch_bindings.cpp — which only
// sees the ops.h declarations — with an unsatisfied reference. They live here
// instead: one GPU translation unit that includes the kernel header and
// defines every entry point exactly once, with external linkage, like the api
// unit does for wvSplitK_int4_g.
#include "skinny_gemms_int4.cuh"

// Carries the public declarations of the entry points defined below, so a
// signature drift is a compile error here rather than a link error later
// (same rule as skinny_gemms_int4_api.cu).
#include "ops.h"

void moe_skinny_int4_decode(
    const at::Tensor& input, const at::Tensor& w13,
    const at::Tensor& w13_scale, const at::Tensor& w2,
    const at::Tensor& w2_scale, const at::Tensor& topk_weights,
    const at::Tensor& topk_ids, at::Tensor& act_buf, at::Tensor& output,
    const int64_t group_size, const std::optional<at::Tensor>& expert_map) {
  const int M = input.size(0);
  const int K = input.size(1);
  const int topk = topk_ids.size(1);
  const int N = act_buf.size(2);

  TORCH_CHECK(M >= 1 && M <= 16, "moe_skinny_int4_decode: M must be 1..16");
  TORCH_CHECK(K % 8 == 0 && N % 8 == 0, "K and N must be divisible by 8");
  TORCH_CHECK(K % group_size == 0 && N % group_size == 0,
              "group_size must divide K and N");
  TORCH_CHECK(input.scalar_type() == at::kHalf &&
                  w13_scale.scalar_type() == at::kHalf &&
                  w2_scale.scalar_type() == at::kHalf &&
                  act_buf.scalar_type() == at::kHalf &&
                  output.scalar_type() == at::kHalf,
              "fp16 activations/scales only");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong,
              "topk_ids must be int32 or int64");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat ||
                  topk_weights.scalar_type() == at::kHalf,
              "topk_weights must be float32 or fp16");
  TORCH_CHECK(topk_ids.is_contiguous() && topk_weights.is_contiguous(),
              "topk_ids/topk_weights must be contiguous");
  const int32_t* emap = nullptr;
  if (expert_map.has_value()) {
    TORCH_CHECK(expert_map->scalar_type() == at::kInt && expert_map->is_contiguous(),
                "expert_map must be contiguous int32");
    emap = reinterpret_cast<const int32_t*>(expert_map->const_data_ptr());
  }
  const bool ids_i64 = topk_ids.scalar_type() == at::kLong;
  const bool w_half = topk_weights.scalar_type() == at::kHalf;
  TORCH_CHECK(input.is_contiguous() && act_buf.is_contiguous() &&
                  output.is_contiguous(),
              "activations must be contiguous");
  const int64_t elem = w13.element_size();
  TORCH_CHECK(w13.numel() * elem == (int64_t)w13_scale.size(0) * 2 * N * K / 2,
              "w13 byte-size mismatch");
  TORCH_CHECK(w2.numel() * elem == (int64_t)w2_scale.size(0) * K * N / 2,
              "w2 byte-size mismatch");
  TORCH_CHECK(output.size(1) == K, "output width must equal hidden size");

  constexpr int WAVES = 8;
  dim3 block(WAVES * 32);
  dim3 grid1((N + WAVES - 1) / WAVES, topk, M);
  dim3 grid2((K + WAVES - 1) / WAVES, 1, M);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  moe_w13_silu_gemv_<WAVES><<<grid1, block, K * 2, stream>>>(
      reinterpret_cast<const half*>(input.const_data_ptr()),
      reinterpret_cast<const uint32_t*>(w13.const_data_ptr()),
      reinterpret_cast<const half*>(w13_scale.const_data_ptr()),
      topk_ids.const_data_ptr(), ids_i64, emap,
      reinterpret_cast<half*>(act_buf.mutable_data_ptr()), K, N, topk,
      (int)group_size);
  moe_w2_gemv_<WAVES><<<grid2, block, topk * N * 2, stream>>>(
      reinterpret_cast<const half*>(act_buf.const_data_ptr()),
      reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
      reinterpret_cast<const half*>(w2_scale.const_data_ptr()),
      topk_ids.const_data_ptr(), ids_i64, emap,
      topk_weights.const_data_ptr(), w_half,
      reinterpret_cast<half*>(output.mutable_data_ptr()), N, K, topk,
      (int)group_size);
}
void moe_decode_v3(const at::Tensor& input, const at::Tensor& topk_weights, const at::Tensor& topk_ids,
                   const at::Tensor& w13, const at::Tensor& w13_scale, const at::Tensor& w2,
                   const at::Tensor& w2_scale, const int64_t group_size,
                   const std::optional<at::Tensor>& expert_map, const at::Tensor& se1, const at::Tensor& se1_s,
                   const at::Tensor& se2, const at::Tensor& se2_s, const at::Tensor& seg,
                   at::Tensor& act_buf, at::Tensor& act_sh, at::Tensor& sgate_buf, at::Tensor& shared_out,
                   at::Tensor& output) {
  const int M = input.size(0), K = input.size(1), N = act_buf.size(2);
  const int I_sh = act_sh.size(1), topk = topk_ids.size(1);
  TORCH_CHECK(M >= 1 && M <= 16 && topk >= 1 && topk <= 32, "moe_decode_v3: M 1..16, topk <= 32");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous() && topk_weights.scalar_type() == at::kFloat &&
                  topk_weights.is_contiguous() && topk_ids.size(0) == M, "moe_decode_v3: topk_ids int32, topk_weights fp32 [M, topk]");
  TORCH_CHECK(K % 32 == 0 && N % 32 == 0 && K <= 4096 && group_size % 32 == 0 && K % group_size == 0 &&
                  N % group_size == 0, "moe_decode_v3: shape constraints");
  TORCH_CHECK(se1.scalar_type() == at::kChar && se1.size(0) == 2 * I_sh && se1.size(1) == K && se1.is_contiguous() &&
                  se2.scalar_type() == at::kChar && se2.size(0) == K && se2.size(1) == I_sh && se2.is_contiguous() &&
                  I_sh % 16 == 0 && I_sh <= 512, "moe_decode_v3: shared expert int8 weights [2I, K], [K, I], I % 16");
  TORCH_CHECK(se1_s.scalar_type() == at::kHalf && se1_s.numel() == 2 * I_sh && se2_s.scalar_type() == at::kHalf &&
                  se2_s.numel() == K && seg.scalar_type() == at::kHalf && seg.numel() == K && seg.is_contiguous(),
              "moe_decode_v3: shared expert scales / gate");
  TORCH_CHECK(input.scalar_type() == at::kHalf && input.is_contiguous() && output.is_contiguous() &&
                  act_buf.is_contiguous() && act_sh.is_contiguous() && sgate_buf.scalar_type() == at::kFloat &&
                  shared_out.is_contiguous() && shared_out.scalar_type() == at::kHalf && shared_out.numel() == output.numel(),
              "moe_decode_v3: buffers");
  for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&input, &act_buf, &act_sh, &w13, &w2, &se1, &se2, &seg})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->const_data_ptr()) % 16 == 0, "moe_decode_v3: 16-byte alignment");
  const int32_t* emap = nullptr;
  if (expert_map.has_value()) {
    TORCH_CHECK(expert_map->scalar_type() == at::kInt && expert_map->is_contiguous(), "expert_map int32");
    emap = reinterpret_cast<const int32_t*>(expert_map->const_data_ptr());
  }
  constexpr int WAVES = 8;
  const int gx = std::max((N + WAVES - 1) / WAVES, (I_sh + WAVES - 1) / WAVES + 1);
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  auto H = [](const at::Tensor& t) { return reinterpret_cast<const half*>(t.const_data_ptr()); };
  moe_w13_v3_<WAVES><<<dim3(gx, topk + 1, M), WAVES * 32, 0, st>>>(
      H(input), topk_ids.data_ptr<int32_t>(), emap, reinterpret_cast<const uint4*>(w13.const_data_ptr()),
      H(w13_scale), reinterpret_cast<half*>(act_buf.mutable_data_ptr()),
      reinterpret_cast<const int8_t*>(se1.const_data_ptr()), H(se1_s), H(seg),
      reinterpret_cast<half*>(act_sh.mutable_data_ptr()), sgate_buf.data_ptr<float>(), I_sh, K, N, topk,
      (int)group_size);
  moe_w2_v3_<WAVES><<<dim3((K + WAVES - 1) / WAVES, 1, M), WAVES * 32, 0, st>>>(
      H(act_buf), reinterpret_cast<const uint4*>(w2.const_data_ptr()), H(w2_scale), topk_ids.data_ptr<int32_t>(),
      topk_weights.data_ptr<float>(), emap, reinterpret_cast<const int8_t*>(se2.const_data_ptr()), H(se2_s), H(act_sh),
      sgate_buf.data_ptr<float>(), I_sh, reinterpret_cast<half*>(shared_out.mutable_data_ptr()),
      reinterpret_cast<half*>(output.mutable_data_ptr()), N, K, topk,
      (int)group_size);
}
void moe_skinny_int4_decode_v2(
    const at::Tensor& input, const at::Tensor& w13, const at::Tensor& w13_scale,
    const at::Tensor& w2, const at::Tensor& w2_scale, const at::Tensor& topk_weights,
    const at::Tensor& topk_ids, at::Tensor& act_buf, at::Tensor& output,
    const int64_t group_size, const std::optional<at::Tensor>& expert_map,
    const std::optional<at::Tensor>& residual) {
  const int M = input.size(0), K = input.size(1);
  const int topk = topk_ids.size(1), N = act_buf.size(2);
  TORCH_CHECK(M >= 1 && M <= 16, "moe_skinny_int4_decode_v2: M must be 1..16");
  TORCH_CHECK(topk <= 32, "moe_skinny_int4_decode_v2: topk <= 32");
  TORCH_CHECK(K % 32 == 0 && N % 32 == 0 && K <= 4096, "moe_skinny_int4_decode_v2: K, N % 32, K <= 4096");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(act_buf.const_data_ptr()) % 16 == 0, "activations must be 16-byte aligned");
  TORCH_CHECK(group_size % 32 == 0 && K % group_size == 0 && N % group_size == 0,
              "moe_skinny_int4_decode_v2: group_size must be a multiple of 32 dividing K and N");
  TORCH_CHECK(input.scalar_type() == at::kHalf && w13_scale.scalar_type() == at::kHalf &&
                  w2_scale.scalar_type() == at::kHalf && act_buf.scalar_type() == at::kHalf &&
                  output.scalar_type() == at::kHalf, "fp16 activations/scales only");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong,
              "topk_ids must be int32 or int64");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat || topk_weights.scalar_type() == at::kHalf,
              "topk_weights must be float32 or fp16");
  TORCH_CHECK(topk_ids.is_contiguous() && topk_weights.is_contiguous() && input.is_contiguous() &&
                  act_buf.is_contiguous() && output.is_contiguous(), "contiguous operands");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w13.const_data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w2.const_data_ptr()) % 16 == 0, "weights must be 16-byte aligned");
  const int64_t elem = w13.element_size();
  TORCH_CHECK(w13.numel() * elem == (int64_t)w13_scale.size(0) * 2 * N * K / 2, "w13 byte-size mismatch");
  TORCH_CHECK(w2.numel() * elem == (int64_t)w2_scale.size(0) * K * N / 2, "w2 byte-size mismatch");
  TORCH_CHECK(output.size(1) == K, "output width must equal hidden size");
  const int32_t* emap = nullptr;
  if (expert_map.has_value()) {
    TORCH_CHECK(expert_map->scalar_type() == at::kInt && expert_map->is_contiguous(),
                "expert_map must be contiguous int32");
    emap = reinterpret_cast<const int32_t*>(expert_map->const_data_ptr());
  }
  const half* rp = nullptr;
  if (residual.has_value()) {
    TORCH_CHECK(residual->scalar_type() == at::kHalf && residual->is_contiguous() &&
                    residual->size(0) == M && residual->size(1) == K, "residual must be contiguous fp16 [M, K]");
    rp = reinterpret_cast<const half*>(residual->const_data_ptr());
  }
  const bool ids_i64 = topk_ids.scalar_type() == at::kLong;
  const bool w_half = topk_weights.scalar_type() == at::kHalf;
  constexpr int WAVES = 8;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  moe_w13_silu_v2_<WAVES><<<dim3((N + WAVES - 1) / WAVES, topk, M), WAVES * 32, 0, stream>>>(
      reinterpret_cast<const half*>(input.const_data_ptr()),
      reinterpret_cast<const uint4*>(w13.const_data_ptr()),
      reinterpret_cast<const half*>(w13_scale.const_data_ptr()), topk_ids.const_data_ptr(), ids_i64,
      emap, reinterpret_cast<half*>(act_buf.mutable_data_ptr()), K, N, topk, (int)group_size);
  moe_w2_v2_<WAVES><<<dim3((K + WAVES - 1) / WAVES, 1, M), WAVES * 32, 0, stream>>>(
      reinterpret_cast<const half*>(act_buf.const_data_ptr()),
      reinterpret_cast<const uint4*>(w2.const_data_ptr()),
      reinterpret_cast<const half*>(w2_scale.const_data_ptr()), topk_ids.const_data_ptr(), ids_i64,
      emap, topk_weights.const_data_ptr(), w_half, rp,
      reinterpret_cast<half*>(output.mutable_data_ptr()), N, K, topk, (int)group_size);
}
at::Tensor gemv_f16_rdna2(const at::Tensor& x, const at::Tensor& w,
                          const std::optional<at::Tensor>& bias) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= 16, "gemv_f16_rdna2: M must be 1..16");
  TORCH_CHECK(w.size(1) == K, "gemv_f16_rdna2: K mismatch");
  TORCH_CHECK(K % 8 == 0, "gemv_f16_rdna2: K % 8 == 0");
  TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf,
              "gemv_f16_rdna2: fp16 only");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
              "gemv_f16_rdna2: contiguous operands");
  const half* bp = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kHalf && bias->is_contiguous() &&
                    bias->numel() == N,
                "gemv_f16_rdna2: bias must be contiguous fp16 [N]");
    bp = reinterpret_cast<const half*>(bias->const_data_ptr());
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto y = at::empty({M, N}, x.options());
  const cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.const_data_ptr());
  const half* wp = reinterpret_cast<const half*>(w.const_data_ptr());
  half* yp = reinterpret_cast<half*>(y.mutable_data_ptr());
  // M = 16 natively (CUDA graphs pad decode batches 9..16 to 16); 9..15 (eager) as an
  // 8-row pass plus a pass over the remaining rows.
  auto run = [&](int m, const half* xs, half* ys) {
    switch (m) {
      case 1: gemv_f16_rdna2_launch<1>(xs, wp, bp, ys, N, K, s); break;
      case 2: gemv_f16_rdna2_launch<2>(xs, wp, bp, ys, N, K, s); break;
      case 3: gemv_f16_rdna2_launch<3>(xs, wp, bp, ys, N, K, s); break;
      case 4: gemv_f16_rdna2_launch<4>(xs, wp, bp, ys, N, K, s); break;
      case 5: gemv_f16_rdna2_launch<5>(xs, wp, bp, ys, N, K, s); break;
      case 6: gemv_f16_rdna2_launch<6>(xs, wp, bp, ys, N, K, s); break;
      case 7: gemv_f16_rdna2_launch<7>(xs, wp, bp, ys, N, K, s); break;
      case 16: gemv_f16_rdna2_launch<16>(xs, wp, bp, ys, N, K, s); break;
      default: gemv_f16_rdna2_launch<8>(xs, wp, bp, ys, N, K, s); break;
    }
  };
  if (M > 8 && M < 16) {
    run(8, xp, yp);
    run(M - 8, xp + (size_t)8 * K, yp + (size_t)8 * N);
  } else {
    run(M, xp, yp);
  }
  return y;
}
at::Tensor gemv_i8_rdna2(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scale,
                         const std::optional<at::Tensor>& bias) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= 16, "gemv_i8_rdna2: M must be 1..16");
  TORCH_CHECK(w.size(1) == K && K % 16 == 0, "gemv_i8_rdna2: K mismatch / K % 16");
  TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kChar &&
                  scale.scalar_type() == at::kHalf && scale.numel() == N,
              "gemv_i8_rdna2: x fp16, w int8, scale fp16[N]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && scale.is_contiguous(),
              "gemv_i8_rdna2: contiguous operands");
  const half* bp = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kHalf && bias->is_contiguous() &&
                    bias->numel() == N, "gemv_i8_rdna2: bias fp16[N]");
    bp = reinterpret_cast<const half*>(bias->const_data_ptr());
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto y = at::empty({M, N}, x.options());
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.const_data_ptr());
  const int8_t* wp = reinterpret_cast<const int8_t*>(w.const_data_ptr());
  const half* sp = reinterpret_cast<const half*>(scale.const_data_ptr());
  half* yp = reinterpret_cast<half*>(y.mutable_data_ptr());
  // M = 16 natively (graph-padded decode batches 9..16); 9..15 as 8 + remainder.
  auto run = [&](int m, const half* xs, half* ys) {
    switch (m) {
      case 1: gemv_i8_rdna2_launch<1>(xs, wp, sp, bp, ys, N, K, st); break;
      case 2: gemv_i8_rdna2_launch<2>(xs, wp, sp, bp, ys, N, K, st); break;
      case 3: gemv_i8_rdna2_launch<3>(xs, wp, sp, bp, ys, N, K, st); break;
      case 4: gemv_i8_rdna2_launch<4>(xs, wp, sp, bp, ys, N, K, st); break;
      case 5: gemv_i8_rdna2_launch<5>(xs, wp, sp, bp, ys, N, K, st); break;
      case 6: gemv_i8_rdna2_launch<6>(xs, wp, sp, bp, ys, N, K, st); break;
      case 7: gemv_i8_rdna2_launch<7>(xs, wp, sp, bp, ys, N, K, st); break;
      case 16: gemv_i8_rdna2_launch<16>(xs, wp, sp, bp, ys, N, K, st); break;
      default: gemv_i8_rdna2_launch<8>(xs, wp, sp, bp, ys, N, K, st); break;
    }
  };
  if (M > 8 && M < 16) {
    run(8, xp, yp);
    run(M - 8, xp + (size_t)8 * K, yp + (size_t)8 * N);
  } else {
    run(M, xp, yp);
  }
  return y;
}
