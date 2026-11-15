// The K largest of a row, as indices in index order, ties to the lower index: a threadgroup a row.
// A long row (DSV41_TOPK_FAST_MIN keys or more) first reads two passes instead of six: a threshold from a strided
// sample (the key whose sample rank stands for sqrt(K * CAP) keys of the row, midway in ratio between too few and too
// many), one pass counting the keys at or above it, and, where they are at least K and fit the threadgroup's
// DSV41_TOPK_CAP, one pass writing them in index order; the select and the emit then run over those survivors, which
// hold every key the full select would pick (at least K keys stand at or above the threshold, so the K-th largest and
// its ties do). Any other count takes the full select below. Either way the picks are the same.
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int l = threadgroup_position_in_grid.y, n = x_shape[1];
const auto xr = x + size_t(l) * n;
threadgroup atomic_uint hist[256];
threadgroup uint t_sel[2], t_part[TG / 32];
// The sample's keys, then the survivors' keys [0, CAP) and their indices [CAP, 2 CAP).
threadgroup uint s_buf[2 * DSV41_TOPK_CAP];
uint prefix = 0, krem = K;
if (n >= DSV41_TOPK_FAST_MIN && 2 * K <= DSV41_TOPK_CAP) {
  for (int i = int(tid); i < DSV41_TOPK_SAMP; i += TG) s_buf[i] = dsv41_key(xr[size_t(i) * size_t(n) / DSV41_TOPK_SAMP]);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const float want = metal::sqrt(float(K) * float(DSV41_TOPK_CAP)) * float(DSV41_TOPK_SAMP) / float(n);
  uint thr = 0, rank = uint(metal::clamp(want + 0.5f, 1.0f, float(DSV41_TOPK_SAMP)));
  DSV41_TOPK_SELECT(thr, rank, DSV41_TOPK_SAMP, s_buf[i])
  // Contiguous index chunks a thread: the survivors before this chunk, and all of them.
  const int C = (n + TG - 1) / TG, i0 = int(tid) * C, i1 = metal::min(n, i0 + C);
  uint ge = 0;
  for (int i = i0; i < i1; ++i) ge += dsv41_key(xr[i]) >= thr;
  const uint s = metal::simd_prefix_exclusive_sum(ge);
  if (lane == 31) t_part[sg] = s + ge;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  uint off = 0, total = 0;
  for (uint g = 0; g < uint(TG / 32); ++g) {
    const uint v = t_part[g];
    off += g < sg ? v : 0u;
    total += v;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (total >= uint(K) && total <= uint(DSV41_TOPK_CAP)) {
    uint p = s + off;
    for (int i = i0; i < i1; ++i) {
      const uint key = dsv41_key(xr[i]);
      if (key >= thr) {
        s_buf[p] = key;
        s_buf[DSV41_TOPK_CAP + p] = uint(i);
        ++p;
      }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    DSV41_TOPK_SELECT(prefix, krem, total, s_buf[i])
    DSV41_TOPK_EMIT(prefix, krem, total, s_buf[i], int(s_buf[DSV41_TOPK_CAP + i]))
    return;
  }
}
DSV41_TOPK_SELECT(prefix, krem, n, dsv41_key(xr[i]))
DSV41_TOPK_EMIT(prefix, krem, n, dsv41_key(xr[i]), i)
