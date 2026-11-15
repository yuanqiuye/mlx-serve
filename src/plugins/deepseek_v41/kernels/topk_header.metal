inline uint dsv41_key(float f) {
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// The fast path's survivors a threadgroup holds (a key and an index each), and its sample, which fills the same words.
#define DSV41_TOPK_CAP 2048
#define DSV41_TOPK_SAMP 4096
#define DSV41_TOPK_FAST_MIN 32768
// An 8-bit radix select over `count` keys (`KEYEXPR` of i): on return `prefix` is the key of rank `krem` from the top
// and `krem` how many of the keys equal to it the rank still takes. Bins 255..0 sit 8 a lane of simdgroup 0: the digit
// is where the count from the top reaches `krem`.
#define DSV41_TOPK_SELECT(prefix, krem, count, KEYEXPR) \
  for (int pass = 0; pass < 4; ++pass) { \
    const int shift = 24 - 8 * pass; \
    const uint hmask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8)); \
    for (uint b = tid; b < 256; b += TG) atomic_store_explicit(&hist[b], 0u, metal::memory_order_relaxed); \
    threadgroup_barrier(metal::mem_flags::mem_threadgroup); \
    for (int i = int(tid); i < int(count); i += TG) { \
      const uint key = (KEYEXPR); \
      if ((key & hmask) == prefix) atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, metal::memory_order_relaxed); \
    } \
    threadgroup_barrier(metal::mem_flags::mem_threadgroup); \
    if (sg == 0) { \
      uint c[8], tot = 0; \
      for (int j = 0; j < 8; ++j) { c[j] = atomic_load_explicit(&hist[255 - (lane * 8 + j)], metal::memory_order_relaxed); tot += c[j]; } \
      const uint before = metal::simd_prefix_exclusive_sum(tot); \
      const bool mine = before < krem && before + tot >= krem; \
      if (mine) { \
        uint acc = before; \
        int j = 0; \
        for (; j < 8; ++j) { if (acc + c[j] >= krem) break; acc += c[j]; } \
        t_sel[0] = prefix | (uint(255 - (lane * 8 + j)) << shift); \
        t_sel[1] = krem - acc; \
      } \
    } \
    threadgroup_barrier(metal::mem_flags::mem_threadgroup); \
    prefix = t_sel[0]; \
    krem = t_sel[1]; \
    threadgroup_barrier(metal::mem_flags::mem_threadgroup); \
  }
// The picks of `count` keys in their order (`IDXEXPR` of i the index a key stands for): every key above `prefix` and
// the first `krem` equal to it. Contiguous chunks a thread, exclusive scans across the threadgroup (ties before a
// chunk, then picks before it), so the output keeps index order.
#define DSV41_TOPK_EMIT(prefix, krem, count, KEYEXPR, IDXEXPR) { \
  const int C = (int(count) + TG - 1) / TG, i0 = int(tid) * C, i1 = metal::min(int(count), i0 + C); \
  uint gt = 0, eq = 0; \
  for (int i = i0; i < i1; ++i) { const uint key = (KEYEXPR); gt += key > prefix; eq += key == prefix; } \
  uint s = metal::simd_prefix_exclusive_sum(eq); \
  if (lane == 31) t_part[sg] = s + eq; \
  threadgroup_barrier(metal::mem_flags::mem_threadgroup); \
  uint off = 0; \
  for (uint g = 0; g < sg; ++g) off += t_part[g]; \
  const uint eq_before = s + off; \
  const uint take = eq_before >= krem ? 0u : metal::min(eq, krem - eq_before); \
  const uint sel = gt + take; \
  threadgroup_barrier(metal::mem_flags::mem_threadgroup); \
  s = metal::simd_prefix_exclusive_sum(sel); \
  if (lane == 31) t_part[sg] = s + sel; \
  threadgroup_barrier(metal::mem_flags::mem_threadgroup); \
  off = 0; \
  for (uint g = 0; g < sg; ++g) off += t_part[g]; \
  uint pos = s + off, taken = 0; \
  for (int i = i0; i < i1; ++i) { \
    const uint key = (KEYEXPR); \
    if (key > prefix || (key == prefix && taken < take)) { \
      taken += key == prefix; \
      out[size_t(l) * K + pos++] = (IDXEXPR); \
    } \
  } }
