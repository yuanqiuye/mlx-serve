//! A prompt's Engram rows read once per table and past the page cache.
//!
//! The pack's tables (oMLX's lossless layout, ~100 GB a layer) keep a row's e4m3 codes (256 B) and its e8m0 scales
//! (8 B) in two regions of one file. Stock, a prompt grid's rows are read span by span (256 tokens: 6,144 ids a table),
//! each span's distinct rows by two preads a row through the page cache: a 16K-token prompt's ~587K row reads cost ~522K
//! 16 KiB page-ins (~8 GiB of file cache, ~1.8 s for the first table, layer 1 waiting ~1.1 s with the pack's files
//! evicted: en1, w-pio-20261101-1419). The page cache absorbs the rows that repeat across spans (~47%) and the scale
//! pages shared by neighbouring rows; read past it one pread at a time, the same reads cost a 4 KiB device IO each
//! (en1's direct arm: 1.17M IOs, 8% slower).
//! Beside a streamed bank (`Table.readDirect`; a resident pack keeps the page cache, which serves a warm prompt's rows
//! faster), a prompt grid's table is read whole: every span's ids sorted and made distinct, the distinct rows read past the
//! cache (F_NOCACHE: the file system blocks that hold them, fstatfs `f_bsize`, APFS 4,096 B, straight into an aligned
//! buffer) on up to 64 threads, each a contiguous run of the sorted rows that keeps its last scale block for the rows
//! after it, then each span's rows copied out of them; a decode token's rows read past the cache as before, a pread a
//! half-row. A repacked table (`engram/engram-manifest.json`: a row's codes then its scales in one 264 B record) is
//! read a record a row (`readRecord`): one pread through the blocks that hold it (two blocks for the 6.4% of records
//! that straddle one), in a whole-table read and a decode token's alike. Exact: the same bytes land in the same rows;
//! only how they are read changes.

const std = @import("std");
const builtin = @import("builtin");

/// The most bytes one read through the blocks may span: a longer row reads with one pread as before.
pub const max_span: usize = 16384;

/// The tables' files read through their blocks (`register`): fd and block.
const max_files = 8;
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    files: [max_files]struct { fd: c_int, block: u64 } = @splat(.{ .fd = -1, .block = 0 }),
    n_files: usize = 0,
    /// The tables' files mapped (`map`).
    maps: [max_files]Map = @splat(.{}),
    n_maps: usize = 0,
};

/// A mapped table file. A row is copied out of the map only from pages the page cache holds: a page it does not hold
/// is read by a page fault, and faults on one file run one after another (a never-read prompt's rows took six times a
/// pread's time). `held` is the kernel's answer when it was last asked (`refresh`), plus every page a pread has read
/// since.
const Map = struct {
    fd: c_int = -1,
    base: [*]const u8 = undefined,
    len: usize = 0,
    /// A byte a page, as `mincore` fills it: bit 0 set while the page cache is taken to hold the page.
    held: []u8 = &.{},
    /// log2 of the page size.
    shift: u6 = 14,
    ctl: ?*Ctl = null,

    fn holds(m: *const Map, off: u64, n: usize) bool {
        const a: usize = @intCast(off >> m.shift);
        const b: usize = @intCast((off + n - 1) >> m.shift);
        return @atomicLoad(u8, &m.held[a], .monotonic) & @atomicLoad(u8, &m.held[b], .monotonic) & 1 != 0;
    }

    fn mark(m: *const Map, off: u64, n: usize) void {
        @atomicStore(u8, &m.held[@intCast(off >> m.shift)], 1, .monotonic);
        @atomicStore(u8, &m.held[@intCast((off + n - 1) >> m.shift)], 1, .monotonic);
    }

    /// `dst` out of the map. One copy in eight is timed: the kernel drops pages without a word, and a copy that
    /// waited on the drive says `held` has gone stale.
    fn copy(m: *const Map, dst: []u8, off: u64) void {
        const src = m.base[@intCast(off)..][0..dst.len];
        if ((off >> 8) & 7 != 0) return @memcpy(dst, src);
        const t0 = nowNs();
        @memcpy(dst, src);
        if (m.ctl.?.note(nowNs() - t0 > Ctl.slow_ns)) m.distrust();
    }

    /// `held` no longer says what the page cache holds: nothing is taken as held (every row a pread, which marks its
    /// pages again) until the kernel has been asked anew, at once unless it was asked a moment ago (a table the
    /// page cache cannot hold whole goes stale as fast as it is read; `Ctl.miss` asks again).
    fn distrust(m: *const Map) void {
        for (m.held) |*h| @atomicStore(u8, h, 0, .monotonic);
        if (nowNs() -% m.ctl.?.asked_ns.load(.monotonic) > Ctl.distrust_gap_ns) refresh(m);
    }
};

/// A map's counters, at an address of their own: a refresh in flight holds it.
const Ctl = struct {
    /// A copy longer than this waited on the drive: a copy from a held page takes under 10 us, a page fault that
    /// reads the drive 80 us and up.
    const slow_ns: u64 = 50_000;
    /// Timed copies a verdict is taken over, and the slow ones among them that make `held` stale. A thread that lost
    /// its core mid-copy reads slow too, a few in a window.
    const window: u32 = 2048;
    const stale_at: u32 = 32;
    /// Preads between two looks at the clock, and how old an answer may get while rows are still read by pread.
    const miss_stride: u32 = 4096;
    const refresh_ns: u64 = 10 * std.time.ns_per_s;
    /// The least time between two questions that stale copies ask for.
    const distrust_gap_ns: u64 = 3 * std.time.ns_per_s;

    timed: std.atomic.Value(u32) = .init(0),
    slow: std.atomic.Value(u32) = .init(0),
    misses: std.atomic.Value(u32) = .init(0),
    refreshing: std.atomic.Value(bool) = .init(false),
    /// When the kernel was last asked (ns of the uptime clock).
    asked_ns: std.atomic.Value(u64) = .init(0),

    /// A timed copy; true when the window it closes holds `stale_at` slow copies.
    fn note(c: *Ctl, slow: bool) bool {
        if (slow) _ = c.slow.fetchAdd(1, .monotonic);
        if (c.timed.fetchAdd(1, .monotonic) + 1 < window) return false;
        c.timed.store(0, .monotonic);
        return c.slow.swap(0, .monotonic) >= stale_at;
    }

    /// A pread of a page not taken as held; true when it is time to ask the kernel again.
    fn miss(c: *Ctl) bool {
        if (c.misses.fetchAdd(1, .monotonic) % miss_stride != 0) return false;
        return nowNs() -% c.asked_ns.load(.monotonic) > refresh_ns;
    }
};

extern "c" fn clock_gettime_nsec_np(clock_id: c_int) u64;
extern "c" fn mincore(addr: *const anyopaque, len: usize, vec: [*]u8) c_int;
extern "c" fn usleep(usec: c_uint) c_int;

/// The uptime clock in ns (a file is mapped on Darwin only).
fn nowNs() u64 {
    return if (comptime builtin.os.tag.isDarwin()) clock_gettime_nsec_np(8) else 0;
}

/// The kernel asked which of `m`'s pages the page cache holds, on a thread of its own (a second or two a 100 GB
/// table); a request while one is in flight is dropped.
fn refresh(m: *const Map) void {
    const ctl = m.ctl.?;
    if (ctl.refreshing.swap(true, .acq_rel)) return;
    ctl.asked_ns.store(nowNs(), .monotonic);
    const th = std.Thread.spawn(.{}, ask, .{ m.base, m.len, m.held, ctl }) catch {
        ctl.refreshing.store(false, .release);
        return;
    };
    th.detach();
}

fn ask(base: [*]const u8, len: usize, held: []u8, ctl: *Ctl) void {
    // A failed call leaves `held` as it was: zeros are preads, and a stale byte is found by the timed copies.
    if (comptime builtin.os.tag.isDarwin()) _ = mincore(base, len, held.ptr);
    ctl.refreshing.store(false, .release);
}

fn waitRefresh(ctl: *const Ctl) void {
    if (comptime !builtin.os.tag.isDarwin()) return;
    while (ctl.refreshing.load(.acquire)) _ = usleep(1000);
}
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_engram_direct", .{});
pub inline fn ps() *State {
    return shared_state.get();
}

/// At a table's open: the file set F_NOCACHE and its block kept, when the block is a power of two
/// in [512, 4096] (else the file stays on the page cache).
pub fn register(fd: c_int) void {
    if (comptime !builtin.os.tag.isDarwin()) return;
    if (ps().n_files == max_files) return;
    var st: Statfs = undefined;
    if (fstatfs(fd, &st) != 0) return;
    const b: u64 = st.f_bsize;
    if (b < 512 or b > 4096 or b & (b - 1) != 0) return;
    if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) == -1) return;
    ps().files[ps().n_files] = .{ .fd = fd, .block = b };
    ps().n_files += 1;
}
/// Darwin's `struct statfs` (64-bit inodes, arm64's only layout): only `f_bsize` is read.
const Statfs = extern struct { f_bsize: u32, f_iosize: i32, rest: [2160]u8 };
extern "c" fn fstatfs(fd: c_int, buf: *Statfs) c_int;
comptime {
    if (@sizeOf(Statfs) != 2168) @compileError("engram_direct: struct statfs is 2,168 bytes on Darwin");
}

/// At a table's open, for a file read through the page cache: the file mapped, so a row the page cache holds is
/// copied out of memory. A pread a row is a system call each, and on one file the calls run one after another
/// whatever the threads (577,000 a second on an M5 Ultra); a copy out of a held page is a hundredth of that. A row
/// the page cache does not hold stays a pread (`Map`). The pages are the page cache's own: the map adds no memory.
/// A registered (F_NOCACHE) file is never mapped.
pub fn map(fd: c_int) void {
    if (comptime !builtin.os.tag.isDarwin()) return;
    if (blockOf(fd) != 0 or mapOf(fd) != null or ps().n_maps == max_files) return;
    const size = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (size <= 0) return;
    const page = std.heap.pageSize();
    const held = std.heap.page_allocator.alloc(u8, (@as(usize, @intCast(size)) + page - 1) / page) catch return;
    @memset(held, 0);
    const ctl = std.heap.page_allocator.create(Ctl) catch return std.heap.page_allocator.free(held);
    ctl.* = .{};
    const m = std.posix.mmap(null, @intCast(size), .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0) catch {
        std.heap.page_allocator.destroy(ctl);
        return std.heap.page_allocator.free(held);
    };
    // Rows are read one here, one there: no read-ahead around a fault.
    _ = std.c.madvise(@ptrCast(m.ptr), m.len, std.c.MADV.RANDOM);
    const slot = &ps().maps[ps().n_maps];
    slot.* = .{ .fd = fd, .base = m.ptr, .len = m.len, .held = held, .shift = @intCast(std.math.log2_int(usize, page)), .ctl = ctl };
    ps().n_maps += 1;
    refresh(slot);
}

/// The file's map let go of (its close, or a table set to read past the page cache).
pub fn unmap(fd: c_int) void {
    if (comptime !builtin.os.tag.isDarwin()) return;
    var i: usize = 0;
    while (i < ps().n_maps) {
        if (ps().maps[i].fd == fd) {
            const m = ps().maps[i];
            // A refresh in flight writes `held` and reads the map's pages' state: it ends first.
            waitRefresh(m.ctl.?);
            std.posix.munmap(@alignCast(@constCast(m.base[0..m.len])));
            std.heap.page_allocator.free(m.held);
            std.heap.page_allocator.destroy(m.ctl.?);
            ps().maps[i] = ps().maps[ps().n_maps - 1];
            ps().n_maps -= 1;
        } else i += 1;
    }
}

fn mapOf(fd: c_int) ?*const Map {
    for (ps().maps[0..ps().n_maps]) |*m| if (m.fd == fd) return m;
    return null;
}

/// Whether `fd` is mapped: its rows the page cache holds are copies out of the map.
pub fn mapped(fd: c_int) bool {
    return mapOf(fd) != null;
}

/// Test seams. `settle`: the refresh in flight has ended. `forget`: no page of `fd` is taken as held. `holds`:
/// whether the bytes are.
pub fn settle(fd: c_int) void {
    const m = mapOf(fd) orelse return;
    waitRefresh(m.ctl.?);
}
pub fn forget(fd: c_int) void {
    const m = mapOf(fd) orelse return;
    for (m.held) |*h| @atomicStore(u8, h, 0, .monotonic);
}
pub fn holds(fd: c_int, off: u64, n: usize) bool {
    const m = mapOf(fd) orelse return false;
    return m.holds(off, n);
}

/// At a table's close: its file let go of (a later file may take the same descriptor).
pub fn unregister(fd: c_int) void {
    var i: usize = 0;
    while (i < ps().n_files) {
        if (ps().files[i].fd == fd) {
            ps().files[i] = ps().files[ps().n_files - 1];
            ps().n_files -= 1;
        } else i += 1;
    }
}

fn blockOf(fd: c_int) u64 {
    for (ps().files[0..ps().n_files]) |f| if (f.fd == fd) return f.block;
    return 0;
}

/// `dst.len` bytes at `off` of `fd`: through the blocks that cover them when `fd` is registered (one F_NOCACHE pread of
/// whole blocks into an aligned buffer, the bytes copied out), else one pread. False on a short read.
pub fn readRow(fd: c_int, dst: []u8, off: u64) bool {
    // Windows opens no Engram table (`engram.Table.open`).
    if (comptime builtin.os.tag == .windows) return false;
    if (comptime builtin.os.tag.isDarwin()) if (mapOf(fd)) |m| {
        if (off > m.len or dst.len > m.len - off) return false;
        if (dst.len == 0) return true;
        if (m.holds(off, dst.len)) {
            m.copy(dst, off);
            return true;
        }
        if (std.c.pread(fd, dst.ptr, dst.len, @intCast(off)) != @as(isize, @intCast(dst.len))) return false;
        // The pread left its pages in the page cache.
        m.mark(off, dst.len);
        if (m.ctl.?.miss()) refresh(m);
        return true;
    };
    const block = blockOf(fd);
    const lo = if (block > 0) off / block * block else off;
    const hi = if (block > 0) std.mem.alignForward(u64, off + dst.len, block) else off + dst.len;
    if (block == 0 or hi - lo > max_span) {
        return std.c.pread(fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
    }
    var buf: [max_span]u8 align(16384) = undefined;
    const n: usize = @intCast(hi - lo);
    const got = std.c.pread(fd, &buf, n, @intCast(lo));
    const skip: usize = @intCast(off - lo);
    if (got < 0 or @as(usize, @intCast(got)) < skip + dst.len) return false;
    @memcpy(dst, buf[skip..][0..dst.len]);
    return true;
}

/// The longest record `readRecord` reads with one call (a head_dim of 4096's codes and scales).
pub const max_record: usize = 4096 + 128;

/// A record's two parts, adjacent at `off` (`w.len` bytes, then `s.len`): one `readRow` of both, split into `w` and
/// `s` (a record past `max_record`: two). False on a short read.
pub fn readRecord(fd: c_int, w: []u8, s: []u8, off: u64) bool {
    const n = w.len + s.len;
    if (n > max_record) return readRow(fd, w, off) and readRow(fd, s, off + w.len);
    var rec: [max_record]u8 = undefined;
    if (!readRow(fd, rec[0..n], off)) return false;
    @memcpy(w, rec[0..w.len]);
    @memcpy(s, rec[w.len..n]);
    return true;
}

/// One thread's last read through a file's blocks: the next row whose bytes it covers is copied from it, no read.
pub const Block = struct {
    lo: u64 = std.math.maxInt(u64),
    len: usize = 0,
    buf: [max_span]u8 align(16384) = undefined,
    /// `dst.len` bytes at `off` of `fd`: from the kept block when it covers them, else read through the blocks that
    /// cover them (kept for the next row); one pread when `fd` is not registered or the span is past `max_span`.
    pub fn read(b: *Block, fd: c_int, dst: []u8, off: u64) bool {
        if (comptime builtin.os.tag == .windows) return false;
        const block = blockOf(fd);
        if (block == 0) return readRow(fd, dst, off);
        if (b.lo != std.math.maxInt(u64) and off >= b.lo and off + dst.len <= b.lo + b.len) {
            const at: usize = @intCast(off - b.lo);
            @memcpy(dst, b.buf[at..][0..dst.len]);
            return true;
        }
        const lo = off / block * block;
        const hi = std.mem.alignForward(u64, off + dst.len, block);
        if (hi - lo > max_span) return readRow(fd, dst, off);
        const n: usize = @intCast(hi - lo);
        b.lo = std.math.maxInt(u64);
        const got = std.c.pread(fd, &b.buf, n, @intCast(lo));
        const skip: usize = @intCast(off - lo);
        if (got < 0 or @as(usize, @intCast(got)) < skip + dst.len) return false;
        b.lo = lo;
        b.len = @intCast(got);
        @memcpy(dst, b.buf[skip..][0..dst.len]);
        return true;
    }
};

test "engram_direct readRecord: a record's codes then scales in one read, split; past the end a short read" {
    const a = std.testing.allocator;
    var data: [3 * 264]u8 = undefined;
    for (&data, 0..) |*x, i| x.* = @truncate(i *% 7 +% 3);
    const path = try std.fmt.allocPrintSentinel(a, "/tmp/dsv41-engram-records-{d}.bin", .{std.c.getpid()}, 0);
    defer a.free(path);
    const out = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(out >= 0);
    const wrote = std.c.write(out, &data, data.len);
    _ = std.c.close(out);
    defer _ = std.c.unlink(path);
    try std.testing.expectEqual(@as(isize, data.len), wrote);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    var cw: [256]u8 = undefined;
    var cs: [8]u8 = undefined;
    try std.testing.expect(readRecord(fd, &cw, &cs, 264));
    try std.testing.expectEqualSlices(u8, data[264..520], &cw);
    try std.testing.expectEqualSlices(u8, data[520..528], &cs);
    try std.testing.expect(!readRecord(fd, &cw, &cs, 2 * 264 + 100));
}

fn testFile(a: std.mem.Allocator, comptime tag: []const u8, data: []const u8) !struct { fd: c_int, path: [:0]u8 } {
    const path = try std.fmt.allocPrintSentinel(a, "/tmp/dsv41-engram-" ++ tag ++ "-{d}.bin", .{std.c.getpid()}, 0);
    errdefer a.free(path);
    const out = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(out >= 0);
    const wrote = std.c.write(out, data.ptr, data.len);
    _ = std.c.close(out);
    try std.testing.expectEqual(@as(isize, @intCast(data.len)), wrote);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    return .{ .fd = fd, .path = path };
}

test "engram_direct map: a row is a pread until the page cache is known to hold its pages, then a copy of the same bytes" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const page = std.heap.pageSize();
    const data = try a.alloc(u8, 3 * page + 100);
    defer a.free(data);
    for (data, 0..) |*x, i| x.* = @truncate(i *% 31 +% (i >> 9));
    const f = try testFile(a, "map", data);
    defer a.free(f.path);
    defer _ = std.c.unlink(f.path);
    defer _ = std.c.close(f.fd);
    map(f.fd);
    defer unmap(f.fd);
    try std.testing.expect(mapped(f.fd));
    // The kernel's answer: the file just written sits in the page cache.
    settle(f.fd);
    try std.testing.expect(holds(f.fd, 0, data.len));
    var row: [256]u8 = undefined;
    try std.testing.expect(readRow(f.fd, &row, page + 40));
    try std.testing.expectEqualSlices(u8, data[page + 40 ..][0..256], &row);
    // Nothing taken as held: the row is a pread, which marks the pages it read and no other.
    forget(f.fd);
    const straddle = 2 * page - 100;
    try std.testing.expect(!holds(f.fd, straddle, 256));
    try std.testing.expect(readRow(f.fd, &row, straddle));
    try std.testing.expectEqualSlices(u8, data[straddle..][0..256], &row);
    try std.testing.expect(holds(f.fd, straddle, 256));
    try std.testing.expect(!holds(f.fd, 0, 16));
    // Held now: the copy out of the map gives the same bytes.
    @memset(&row, 0);
    try std.testing.expect(readRow(f.fd, &row, straddle));
    try std.testing.expectEqualSlices(u8, data[straddle..][0..256], &row);
    // Past the end: a short read either way.
    try std.testing.expect(!readRow(f.fd, &row, data.len - 100));
}

test "engram_direct map: copies that waited on the drive make the held pages stale; a few slow ones do not" {
    var c: Ctl = .{};
    var stale = false;
    for (0..Ctl.window) |i| stale = c.note(i % 400 == 0) or stale;
    try std.testing.expect(!stale);
    for (0..Ctl.window) |i| stale = c.note(i % 50 == 0) or stale;
    try std.testing.expect(stale);
    // The verdict cleared the window.
    try std.testing.expectEqual(@as(u32, 0), c.slow.load(.monotonic));
}
