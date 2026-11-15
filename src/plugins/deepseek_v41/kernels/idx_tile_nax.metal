// The wide indexer's scores (idx_tile.metal's sums) on the tensor units, after oMLX's
// patches/glm_moe_dsa/indexer_nax.py (Apache-2.0, see NOTICE). A simdgroup owns 32 query rows x 32 keys and takes them
// as two fragments of 16 rows, one after the other, in MLX's NAX fragment layout: a lane holds rows fm and fm + 8 and
// columns fn .. fn + 3 of each 16 x 16 half. Two heads at a time contract their [16, IHD] x [IHD, 32] products against
// one load of the keys, accumulating in f32 on the matrix op. Relu, the row's head weight and the head sum follow in
// f32 in idx_tile.metal's order: heads in pairs, a group of eight as (p01 + p23) + (p45 + p67), the groups added in
// turn. The index queries and keys are fp4 round trips, whose dot products are exact in f32, so the scores are
// idx_tile.metal's bits. One fragment at a time keeps a lane's live sums to 48 floats: with both live the kernel ran
// at half the rate. A key a row does not see yet scores -inf, and a block none of its rows sees skips the products.
constexpr int D = IHD;
constexpr int H = IH;
const int S = q_shape[0];
const int P = keys_shape[0];
const int p0 = meta[0];
const int ratio = meta[1];
const float wscale = wsc[0];
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int sm = int(threadgroup_position_in_grid.y) * 64 + int(sg >> 1) * 32;
const int sn = int(threadgroup_position_in_grid.x) * 64 + int(sg & 1) * 32;

const short qid = short(lane >> 2);
const short fm = short((qid & 4) | ((lane >> 1) & 3));
const short fn = short(((qid & 2) | (lane & 1)) * 4);

const int s_last = metal::min(sm + 31, S - 1);
const bool live = sm < S && sn < P && sn < (p0 + s_last + 1) / ratio;

int qrow[2][2];
for (short tm = 0; tm < 2; ++tm)
  for (short i = 0; i < 2; ++i) qrow[tm][i] = metal::min(sm + tm * 16 + fm + i * 8, S - 1);

const device T* kbase[2][2];
for (short tn = 0; tn < 2; ++tn)
  for (short i = 0; i < 2; ++i) kbase[tn][i] = keys + size_t(metal::min(sn + tn * 16 + fm + i * 8, P - 1)) * D + fn;

for (short tm = 0; tm < 2; ++tm) {
  float acc[16];
  for (short e = 0; e < 16; ++e) acc[e] = 0.0f;
  if (live) {
    constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    const device T* q0 = q + size_t(qrow[tm][0]) * (H * D) + fn;
    const device T* q1 = q + size_t(qrow[tm][1]) * (H * D) + fn;
    for (int g = 0; g < H; g += 8) {
      float qa[16], qb[16];
      #pragma clang loop unroll(full)
      for (short pr = 0; pr < 4; ++pr) {
        const int h = g + pr * 2;
        auto ct_e = op.template get_left_input_cooperative_tensor<T, T, float>();
        auto ct_o = op.template get_left_input_cooperative_tensor<T, T, float>();
        auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
        auto ce = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_e)>, metal::remove_addrspace_t<decltype(ct_b)>, float>();
        auto co = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_e)>, metal::remove_addrspace_t<decltype(ct_b)>, float>();
        for (short e = 0; e < 16; ++e) {
          ce[e] = 0.0f;
          co[e] = 0.0f;
        }
        const device T* e0 = q0 + h * D;
        const device T* e1 = q1 + h * D;
        for (short kk = 0; kk < D; kk += 16) {
          for (short j = 0; j < 4; ++j) {
            ct_e[j] = e0[kk + j];
            ct_e[4 + j] = e1[kk + j];
            ct_o[j] = e0[D + kk + j];
            ct_o[4 + j] = e1[D + kk + j];
          }
          for (short tn = 0; tn < 2; ++tn)
            for (short i = 0; i < 2; ++i)
              for (short j = 0; j < 4; ++j) ct_b[tn * 8 + i * 4 + j] = kbase[tn][i][kk + j];
          op.run(ct_e, ct_b, ce);
          op.run(ct_o, ct_b, co);
        }
        for (short i = 0; i < 2; ++i) {
          const size_t wrow = size_t(qrow[tm][i]) * H;
          const float we = float(w[wrow + h]) * wscale, wo = float(w[wrow + h + 1]) * wscale;
          for (short tn = 0; tn < 2; ++tn)
            for (short j = 0; j < 4; ++j) {
              const short e = tn * 8 + i * 4 + j;
              const float p = metal::max(float(ce[e]), 0.0f) * we + metal::max(float(co[e]), 0.0f) * wo;
              if (pr == 0) qa[e] = p;
              else if (pr == 1) qa[e] += p;
              else if (pr == 2) qb[e] = p;
              else acc[e] += qa[e] + (qb[e] + p);
            }
        }
      }
    }
  }
  for (short i = 0; i < 2; ++i) {
    const int s = sm + tm * 16 + fm + i * 8;
    if (s >= S) continue;
    const int seen = (p0 + s + 1) / ratio;
    for (short tn = 0; tn < 2; ++tn)
      for (short j = 0; j < 4; ++j) {
        const int p = sn + tn * 16 + fn + j;
        if (p >= P) continue;
        scores[size_t(s) * P + p] = p < seen ? acc[tn * 8 + i * 4 + j] : -INFINITY;
      }
  }
}
