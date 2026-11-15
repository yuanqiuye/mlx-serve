//! A wide prompt layer's quantized linears dequantized once a prompt, not once a chunk.
//!
//! From `DENSE_ROWS` rows a quantized linear dequantizes its weight and runs the dense GEMM (`Lin.apply`). The
//! dequantized weight was the chunk's own: every chunk of a long prompt dequantized every wide linear again, into
//! buffers MLX allocated again (a Metal System Trace of a 41K prompt: `mxfp8_dequantize` 7.1% of the prompt's GPU
//! time at 4096-row calls, about 64 ms a call set, beside the host's ~15 ms a GB for the 5 GB of new buffers).
//!
//! So the first chunk's dequantized weight is kept by an extra reference, keyed on the quantized weight's own handle
//! and the dtype asked, and the next chunks' linears read it. The same dequantize op's output over the same codes:
//! the same bits, read from a buffer that already holds them.
//!
//! Memory: the trunk's wide linears a layer (about 128M weights, 256 MB in bf16) over the layers the encoder rows
//! run, about 5 GB, held from a prompt's first chunk to the first decode-sized forward after it (`releaseAll`, where
//! the prompt's scratch regions go too), capped at `max_bytes`. A decode step never reads them: its rows run the
//! quantized kernels.
//!
//! `MLX_SERVE_DSV41_DENSE_KEEP=0` dequantizes every chunk as before.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");

const A = mlx.mlx_array;

/// Dequantized weights kept at most: a layer's wide linears times the trunk's layers, with room.
pub const max_kept = 640;
/// Bytes kept at most; past it a linear dequantizes for its chunk alone, as before.
pub const max_bytes: u64 = 12 << 30;

const Entry = struct { key: usize, dt: mlx.mlx_dtype, d: A };

/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    items: [max_kept]Entry = undefined,
    n: usize = 0,
    bytes: u64 = 0,
    on: ?bool = null,
};
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_dense_keep", .{});
pub inline fn ps() *State {
    return shared_state.get();
}

pub fn enabled() bool {
    const st = ps();
    if (st.on == null) st.on = if (env.get(.MLX_SERVE_DSV41_DENSE_KEEP)) |v| v[0] != '0' else true;
    return st.on.?;
}

fn bytesOf(x: A) u64 {
    return @as(u64, mlx.mlx_array_size(x)) * @as(u64, mlx.mlx_array_itemsize(x));
}

/// The kept dequantized weight of the quantized weight `w` in `dt`, or null. The cache's own handle: the caller
/// reads it and frees nothing.
pub fn get(w: A, dt: mlx.mlx_dtype) ?A {
    const st = ps();
    const key = @intFromPtr(w.ctx);
    for (st.items[0..st.n]) |e| {
        if (e.key == key and e.dt == dt) return e.d;
    }
    return null;
}

/// Keeps `d`, the weight `w` dequantized to `dt` (evaluated or not: the chunk's own graph evaluates it), by a
/// reference of the cache's own. Nothing is kept when the switch is off, the table or the byte cap is full, or `w`
/// is kept already.
pub fn put(w: A, dt: mlx.mlx_dtype, d: A) void {
    const st = ps();
    if (!enabled() or w.ctx == null or d.ctx == null or st.n == max_kept or get(w, dt) != null) return;
    const bytes = bytesOf(d);
    if (st.bytes + bytes > max_bytes) return;
    var h = mlx.mlx_array_new();
    if (mlx.mlx_array_set(&h, d) != 0) {
        _ = mlx.mlx_array_free(h);
        return;
    }
    st.items[st.n] = .{ .key = @intFromPtr(w.ctx), .dt = dt, .d = h };
    st.n += 1;
    st.bytes += bytes;
}

/// Bytes the kept weights hold.
pub fn keptBytes() u64 {
    return ps().bytes;
}

/// A prompt's end (a decode-sized forward, the model's teardown): every kept weight let go.
pub fn releaseAll() void {
    const st = ps();
    for (st.items[0..st.n]) |e| _ = mlx.mlx_array_free(e.d);
    st.n = 0;
    st.bytes = 0;
}

test "dense_keep: a weight is kept once by its own handle and dtype, read back as the same array, let go together" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    releaseAll();
    ps().on = true;
    defer {
        releaseAll();
        ps().on = null;
    }
    const data = [_]f32{ 1, 2, 3, 4 };
    const w1 = mlx.mlx_array_new_data(&data, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(w1);
    const w2 = mlx.mlx_array_new_data(&data, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(w2);
    const d1 = mlx.mlx_array_new_data(&data, &[_]c_int{ 4, 1 }, 2, .float32);
    try std.testing.expect(get(w1, .bfloat16) == null);
    put(w1, .bfloat16, d1);
    // The cache holds its own reference: the caller's handle goes, the kept array stays readable.
    _ = mlx.mlx_array_free(d1);
    const kept = get(w1, .bfloat16) orelse return error.NotKept;
    try std.testing.expectEqual(@as(usize, 4), mlx.mlx_array_size(kept));
    try std.testing.expect(get(w1, .float16) == null);
    try std.testing.expect(get(w2, .bfloat16) == null);
    try std.testing.expectEqual(@as(u64, 16), keptBytes());
    // A second put of the same weight keeps the first.
    put(w1, .bfloat16, w2);
    try std.testing.expectEqual(kept.ctx, (get(w1, .bfloat16) orelse return error.NotKept).ctx);
    try std.testing.expectEqual(@as(u64, 16), keptBytes());
    releaseAll();
    try std.testing.expect(get(w1, .bfloat16) == null);
    try std.testing.expectEqual(@as(u64, 0), keptBytes());
    // Off: nothing is kept.
    ps().on = false;
    put(w1, .bfloat16, w2);
    try std.testing.expect(get(w1, .bfloat16) == null);
}
