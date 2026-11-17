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
/// is read by a page fault, and faults that read the drive run one after another on one file (35,000 a second on an
/// M5 Ultra whatever the threads; a pread of such a page, 170,000). `held` is the kernel's answer when it was last
/// asked (`refresh`), plus every page a pread has read since, plus the pages a row was copied out of (`touched`: in
/// this process's address space, the pages the kernel drops last, and about 95% of a prompt's row reads). The kernel
/// drops pages without a word: copies that page in from the drive say `held` has gone stale (`Ctl.pagedIn`), and
/// until the kernel has answered again only the touched pages' rows are copies, the others preads (`Trust`).
const Map = struct {
    fd: c_int = -1,
    base: [*]const u8 = undefined,
    len: usize = 0,
    /// A byte a page: `in_cache` while the page cache is taken to hold the page, `touched` once a row was copied out
    /// of it since.
    held: []u8 = &.{},
    /// Where the kernel's answer lands before it is merged into `held` (`ask`); the refresh in flight owns it.
    answer: []u8 = &.{},
    /// log2 of the page size.
    shift: u6 = 14,
    ctl: ?*Ctl = null,
    /// The read of the whole file into the page cache, once asked for (`warm`).
    warming: ?*Warm = null,

    const in_cache: u8 = 1;
    const touched: u8 = 2;

    /// The bits both of the bytes' pages have (a row lies on two at most).
    fn bits(m: *const Map, off: u64, n: usize) u8 {
        const a: usize = @intCast(off >> m.shift);
        const b: usize = @intCast((off + n - 1) >> m.shift);
        return @atomicLoad(u8, &m.held[a], .monotonic) & @atomicLoad(u8, &m.held[b], .monotonic);
    }

    fn holds(m: *const Map, off: u64, n: usize) bool {
        return m.bits(off, n) & in_cache != 0;
    }

    /// `want` set on the bytes' pages; a page that has it is not written.
    fn set(m: *const Map, off: u64, n: usize, want: u8) void {
        for ([2]usize{ @intCast(off >> m.shift), @intCast((off + n - 1) >> m.shift) }) |i| {
            const have = @atomicLoad(u8, &m.held[i], .monotonic);
            if (have & want != want) @atomicStore(u8, &m.held[i], have | want, .monotonic);
        }
    }

    /// A pread read the bytes: the page cache holds their pages (a touched page stays touched).
    fn mark(m: *const Map, off: u64, n: usize) void {
        m.set(off, n, in_cache);
    }

    /// Whether the bytes are copied out of the map: their pages taken as held, as far as `held` is trusted.
    fn copies(m: *const Map, off: u64, n: usize) bool {
        const need: u8 = switch (m.ctl.?.trust.load(.monotonic)) {
            .held => in_cache,
            .touched => touched,
            .none => return false,
        };
        return m.bits(off, n) & need != 0;
    }

    /// `dst` out of the map, its pages touched from here on; every `Ctl.stride` copies the task's page-ins are
    /// counted, and an old answer asked anew behind the copies (a load, another program: the page cache moves while
    /// nothing here reads).
    fn copy(m: *const Map, dst: []u8, off: u64) void {
        @memcpy(dst, m.base[@intCast(off)..][0..dst.len]);
        m.set(off, dst.len, in_cache | touched);
        const ctl = m.ctl.?;
        if (!ctl.strideEnds()) return;
        const ins = ctl.pagedIn(pageIns());
        if (ins >= Ctl.stale_at) return m.doubt(ins);
        if (nowNs() -% ctl.asked_ns.load(.monotonic) > Ctl.refresh_ns) refresh(m);
    }

    /// `held` is stale, `ins` page-ins in a stretch of copies: until the kernel has answered again a row is copied out
    /// of the map only from touched pages, and from none once `Ctl.thrash_at` page-ins fall in a stretch of those (the
    /// answer replaces what `held` takes as held; nothing is cleared here, so an answer in flight is not lost). The
    /// kernel is asked at once unless it was asked a moment ago (a table the page cache cannot hold whole goes stale
    /// as fast as it is read; `Ctl.miss` asks then).
    fn doubt(m: *const Map, ins: u32) void {
        const ctl = m.ctl.?;
        switch (ctl.trust.load(.monotonic)) {
            .held => ctl.trust.store(.touched, .monotonic),
            .touched => if (ins >= Ctl.thrash_at) ctl.trust.store(.none, .monotonic),
            .none => {},
        }
        _ = ctl.doubts.fetchAdd(1, .monotonic);
        if (nowNs() -% ctl.asked_ns.load(.monotonic) > Ctl.doubt_gap_ns) refresh(m);
    }
};

/// How far a map's `held` is trusted: wholly; for touched pages only (copies paged in from the drive: some page it
/// takes as held is not); for none (touched pages' copies paged in as well).
const Trust = enum(u8) { held, touched, none };

/// A map's counters, at an address of their own: a refresh in flight holds it.
const Ctl = struct {
    /// Copies between two looks at the task's page-in count, and the page-ins in one such stretch that make `held`
    /// stale. A page-in is a page read from the drive by a fault, the task's own count of them: a copy from a page the
    /// page cache holds adds none however long it takes (a first touch waits on the address space, not the drive), so
    /// nothing here is timed. About one copy in twenty is a page's first, and a first copy that reads the drive costs
    /// five preads' time: three in a stretch is where preads for the untouched pages break even. Suspicion costs
    /// little (touched pages' rows stay copies, the answer takes a third of a second), other files' page-ins count
    /// too: a handful.
    const stride: u32 = 1024;
    const stale_at: u32 = 8;
    /// The page-ins in a stretch of copies out of touched pages alone that say those are being dropped too (a table
    /// the page cache cannot hold). Past one copy in seventeen a pread a row is faster; the count sits above that and
    /// above the copies in flight when the suspicion began (32 threads a table), which page in after it.
    const thrash_at: u32 = 128;
    /// Preads between two looks at the clock, and how old an answer may get while rows are still read by pread.
    const miss_stride: u32 = 4096;
    const refresh_ns: u64 = 10 * std.time.ns_per_s;
    /// The least time between two questions that stale copies ask for.
    const doubt_gap_ns: u64 = 3 * std.time.ns_per_s;

    copied_n: std.atomic.Value(u64) = .init(0),
    /// The task's page-ins when a stretch of copies last ended.
    page_ins: std.atomic.Value(u32) = .init(0),
    misses: std.atomic.Value(u64) = .init(0),
    /// The questions put to the kernel, and the times copies found `held` stale.
    asks: std.atomic.Value(u32) = .init(0),
    doubts: std.atomic.Value(u32) = .init(0),
    refreshing: std.atomic.Value(bool) = .init(false),
    /// How far `held` is trusted until the kernel's next answer.
    trust: std.atomic.Value(Trust) = .init(.held),
    /// When the kernel was last asked (ns of the uptime clock).
    asked_ns: std.atomic.Value(u64) = .init(0),

    /// A copy; true when it ends a stretch of `stride`.
    fn strideEnds(c: *Ctl) bool {
        return (c.copied_n.fetchAdd(1, .monotonic) + 1) % stride == 0;
    }

    /// The end of a stretch of copies, the task's page-ins now `now`: how many fell in the stretch (other files'
    /// page-ins count too: a refresh they cost is one question to the kernel).
    fn pagedIn(c: *Ctl, now: u32) u32 {
        return now -% c.page_ins.swap(now, .monotonic);
    }

    /// A pread of a page not taken as held; true when it is time to ask the kernel again: `held` under suspicion and
    /// the last question `doubt_gap_ns` old, or any answer `refresh_ns` old.
    fn miss(c: *Ctl) bool {
        if (c.misses.fetchAdd(1, .monotonic) % miss_stride != 0) return false;
        const age = nowNs() -% c.asked_ns.load(.monotonic);
        return age > refresh_ns or (c.trust.load(.monotonic) != .held and age > doubt_gap_ns);
    }
};

extern "c" fn clock_gettime_nsec_np(clock_id: c_int) u64;
extern "c" fn mincore(addr: *const anyopaque, len: usize, vec: [*]u8) c_int;
extern "c" fn usleep(usec: c_uint) c_int;
extern "c" fn task_info(task: u32, flavor: u32, info: *anyopaque, count: *u32) c_int;
extern "c" var mach_task_self_: u32;

/// The task's page-ins so far (`TASK_EVENTS_INFO`), 0 where it cannot be read.
fn pageIns() u32 {
    if (comptime !builtin.os.tag.isDarwin()) return 0;
    var events: [8]i32 = undefined;
    var n: u32 = events.len;
    if (task_info(mach_task_self_, 2, &events, &n) != 0 or n < 2) return 0;
    return @bitCast(events[1]);
}

/// The uptime clock in ns (a file is mapped on Darwin only).
fn nowNs() u64 {
    return if (comptime builtin.os.tag.isDarwin()) clock_gettime_nsec_np(8) else 0;
}

/// Threads the kernel is asked on, each a run of the file's pages (a 74 GB table's 4.5M pages: 1.5 s on one, 0.35 s
/// on eight, 0.25 s on sixteen; rows are copied beside it at their own speed), and the fewest pages a thread is worth.
const ask_threads = 8;
const ask_pages: usize = if (builtin.is_test) 1 else 1 << 16;

/// The kernel asked which of `m`'s pages the page cache holds, on threads of its own; a request while one is in
/// flight is dropped.
fn refresh(m: *const Map) void {
    const ctl = m.ctl.?;
    if (ctl.refreshing.swap(true, .acq_rel)) return;
    ctl.asked_ns.store(nowNs(), .monotonic);
    _ = ctl.asks.fetchAdd(1, .monotonic);
    const th = std.Thread.spawn(.{}, ask, .{ m.*, ctl }) catch {
        ctl.refreshing.store(false, .release);
        return;
    };
    th.detach();
}

/// The kernel's answer for a page (`mincore`'s byte: bit 0 is the page cache holding it, the other bits are not this
/// file's `touched`) merged into what `held` had: held or not as the kernel says, touched only while held.
fn merged(have: u8, got: u8) u8 {
    return if (got & 1 != 0) Map.in_cache | (have & Map.touched) else 0;
}

fn ask(m: Map, ctl: *Ctl) void {
    const Scan = struct {
        m: Map,
        parts: usize,
        ok: std.atomic.Value(bool) = .init(true),
        fn part(s: *@This(), k: usize) void {
            const pages = s.m.held.len;
            const lo = pages * k / s.parts;
            const hi = pages * (k + 1) / s.parts;
            if (hi == lo) return;
            const at = lo << s.m.shift;
            const n = @min(s.m.len, hi << s.m.shift) - at;
            if (comptime builtin.os.tag.isDarwin()) {
                if (mincore(s.m.base + at, n, s.m.answer.ptr + lo) != 0) s.ok.store(false, .release);
            }
        }
    };
    var scan: Scan = .{ .m = m, .parts = std.math.clamp(m.held.len / ask_pages, 1, ask_threads) };
    var threads: [ask_threads]?std.Thread = @splat(null);
    for (threads[1..scan.parts], 1..) |*t, k| t.* = std.Thread.spawn(.{}, Scan.part, .{ &scan, k }) catch null;
    scan.part(0);
    for (threads[1..scan.parts], 1..) |t, k| {
        if (t) |th| th.join() else scan.part(k);
    }
    // A failed call leaves `held` as it was: zeros are preads, and a stale byte is found by the copies' page-ins.
    if (scan.ok.load(.acquire)) {
        for (m.held, m.answer) |*h, got| {
            const have = @atomicLoad(u8, h, .monotonic);
            const now = merged(have, got);
            if (now != have) @atomicStore(u8, h, now, .monotonic);
        }
    }
    // The answer is fresh: page-ins before it are not its copies'.
    ctl.page_ins.store(pageIns(), .monotonic);
    ctl.trust.store(.held, .monotonic);
    ctl.refreshing.store(false, .release);
}

fn waitRefresh(ctl: *const Ctl) void {
    if (comptime !builtin.os.tag.isDarwin()) return;
    while (ctl.refreshing.load(.acquire)) _ = usleep(1000);
}
/// A mapped file read into the page cache from end to end (`warm`): a model's load pushes its tables out of the
/// cache, and until a prompt has read a row's page that row is a pread (586,000 a second on one file whatever the
/// threads, where copies out of held pages run at millions). The pages are the page cache's: the system takes
/// them back when it wants the memory, and while rows are still read by pread the pages it took are read again
/// (`again`), at longer and longer intervals while they keep going (`strikes`: the cache is not holding the file).
const Warm = struct {
    stop: std.atomic.Value(bool) = .init(false),
    /// A pass in flight; its thread, joined by the next pass or the unmap.
    running: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// Passes ended, the bytes they read (pages the page cache held are not read), and when the last one ended.
    passes: std.atomic.Value(u32) = .init(0),
    read: std.atomic.Value(u64) = .init(0),
    ended_ns: std.atomic.Value(u64) = .init(0),
    /// Passes in a row after the first that found a quarter of the file gone again.
    strikes: std.atomic.Value(u32) = .init(0),

    /// Readers a file (an SSD serves several streams faster than one) and the bytes of one read: short enough that
    /// a prompt's own preads of the file, which run one after another with these, wait a fraction of a millisecond.
    const readers = 4;
    const chunk: usize = if (builtin.is_test) 4096 else 1 << 20;
    /// The least time between two passes, four times longer a strike (20 s to 21 min).
    const again_ns: u64 = if (builtin.is_test) 0 else 20 * std.time.ns_per_s;
    const max_strikes = 3;

    /// A pass on a thread of its own, unless one is in flight.
    fn start(w: *Warm, m: Map) void {
        if (w.running.swap(true, .acq_rel)) return;
        if (w.thread) |t| t.join();
        w.thread = std.Thread.spawn(.{}, run, .{ m, w }) catch null;
        if (w.thread == null) w.running.store(false, .release);
    }

    /// A pread found a page gone: another pass, once the last is `again_ns` old.
    fn again(w: *Warm, m: *const Map) void {
        if (w.running.load(.monotonic) or w.passes.load(.monotonic) == 0) return;
        if (nowNs() -% w.ended_ns.load(.monotonic) < again_ns << @intCast(2 * w.strikes.load(.monotonic))) return;
        w.start(m.*);
    }

    fn run(m: Map, w: *Warm) void {
        // What the page cache holds now, not when the file was mapped.
        refresh(&m);
        waitRefresh(m.ctl.?);
        const before = w.read.load(.monotonic);
        var threads: [readers]?std.Thread = @splat(null);
        for (threads[1..], 1..) |*t, k| t.* = std.Thread.spawn(.{}, part, .{ m, w, k }) catch null;
        part(m, w, 0);
        for (threads[1..], 1..) |t, k| {
            if (t) |th| th.join() else part(m, w, k);
        }
        // A pass after the first that reads a quarter of the file again: the next one waits longer.
        if (w.passes.load(.monotonic) > 0) {
            const lost = w.read.load(.monotonic) - before > m.len / 4;
            w.strikes.store(if (lost) @min(w.strikes.load(.monotonic) + 1, max_strikes) else 0, .monotonic);
        }
        w.ended_ns.store(nowNs(), .monotonic);
        _ = w.passes.fetchAdd(1, .release);
        w.running.store(false, .release);
    }

    fn part(m: Map, w: *Warm, k: usize) void {
        const buf = std.heap.page_allocator.alloc(u8, chunk) catch return;
        defer std.heap.page_allocator.free(buf);
        const pages = m.held.len;
        var at = (pages * k / readers) << m.shift;
        const end = @min(m.len, (pages * (k + 1) / readers) << m.shift);
        while (at < end and !w.stop.load(.acquire)) {
            const n = @min(chunk, end - at);
            defer at += n;
            const lo = at >> m.shift;
            const hi = (at + n - 1) >> m.shift;
            var held = true;
            for (m.held[lo .. hi + 1]) |*h| held = held and @atomicLoad(u8, h, .monotonic) & Map.in_cache != 0;
            if (held) continue;
            if (std.c.pread(m.fd, buf.ptr, n, @intCast(at)) != @as(isize, @intCast(n))) return;
            // The pread left its pages in the page cache.
            for (m.held[lo .. hi + 1]) |*h| {
                const have = @atomicLoad(u8, h, .monotonic);
                if (have & Map.in_cache == 0) @atomicStore(u8, h, have | Map.in_cache, .monotonic);
            }
            _ = w.read.fetchAdd(n, .monotonic);
        }
    }
};

/// A mapped file's pages read into the page cache on threads of their own, and read again where the system has
/// taken them back while rows are still read (`Warm`); `unmap` ends it. A file not mapped is left alone.
pub fn warm(fd: c_int) void {
    if (comptime !builtin.os.tag.isDarwin()) return;
    for (ps().maps[0..ps().n_maps]) |*m| if (m.fd == fd and m.warming == null) {
        const w = std.heap.page_allocator.create(Warm) catch return;
        w.* = .{};
        m.warming = w;
        w.start(m.*);
    };
}

/// A warmed file's reads: the passes ended, the bytes read into the page cache so far, whether a pass is in flight
/// and the passes in a row that found a quarter of the file gone again. Null for a file never warmed.
pub fn warmed(fd: c_int) ?struct { passes: u32, bytes: u64, running: bool, strikes: u32 } {
    const m = mapOf(fd) orelse return null;
    const w = m.warming orelse return null;
    return .{ .passes = w.passes.load(.acquire), .bytes = w.read.load(.monotonic), .running = w.running.load(.acquire), .strikes = w.strikes.load(.monotonic) };
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
    const pages = (@as(usize, @intCast(size)) + page - 1) / page;
    const held = std.heap.page_allocator.alloc(u8, pages) catch return;
    @memset(held, 0);
    const answer = std.heap.page_allocator.alloc(u8, pages) catch return std.heap.page_allocator.free(held);
    @memset(answer, 0);
    const ctl = std.heap.page_allocator.create(Ctl) catch {
        std.heap.page_allocator.free(answer);
        return std.heap.page_allocator.free(held);
    };
    ctl.* = .{};
    const m = std.posix.mmap(null, @intCast(size), .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0) catch {
        std.heap.page_allocator.destroy(ctl);
        std.heap.page_allocator.free(answer);
        return std.heap.page_allocator.free(held);
    };
    // Rows are read one here, one there: no read-ahead around a fault.
    _ = std.c.madvise(@ptrCast(m.ptr), m.len, std.c.MADV.RANDOM);
    const slot = &ps().maps[ps().n_maps];
    slot.* = .{ .fd = fd, .base = m.ptr, .len = m.len, .held = held, .answer = answer, .shift = @intCast(std.math.log2_int(usize, page)), .ctl = ctl };
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
            // The read-ahead reads the file and writes `held`: it ends first.
            if (m.warming) |w| {
                w.stop.store(true, .release);
                while (w.running.load(.acquire)) _ = usleep(1000);
                if (w.thread) |t| t.join();
                std.heap.page_allocator.destroy(w);
            }
            // A refresh in flight writes `held` and `answer` and reads the map's pages' state: it ends first.
            waitRefresh(m.ctl.?);
            std.posix.munmap(@alignCast(@constCast(m.base[0..m.len])));
            std.heap.page_allocator.free(m.held);
            std.heap.page_allocator.free(m.answer);
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

/// A mapped file's rows since it was mapped: copies out of the map, preads, the questions put to the kernel and the
/// times copies found the held pages stale. Zeros for a file not mapped.
pub const Counts = struct {
    copies: u64 = 0,
    preads: u64 = 0,
    asked: u32 = 0,
    stale: u32 = 0,

    pub fn add(c: *Counts, o: Counts) void {
        c.copies += o.copies;
        c.preads += o.preads;
        c.asked += o.asked;
        c.stale += o.stale;
    }
};
pub fn counts(fd: c_int) Counts {
    const m = mapOf(fd) orelse return .{};
    const ctl = m.ctl.?;
    return .{ .copies = ctl.copied_n.load(.monotonic), .preads = ctl.misses.load(.monotonic), .asked = ctl.asks.load(.monotonic), .stale = ctl.doubts.load(.monotonic) };
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
        if (m.copies(off, dst.len)) {
            m.copy(dst, off);
            return true;
        }
        if (std.c.pread(fd, dst.ptr, dst.len, @intCast(off)) != @as(isize, @intCast(dst.len))) return false;
        // The pread left its pages in the page cache.
        m.mark(off, dst.len);
        if (m.ctl.?.miss()) refresh(m);
        if (m.warming) |w| w.again(m);
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

test "engram_direct map: a warmed file's rows are copies from the first read on, pages taken back are read again, and a close ends the read" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const page = std.heap.pageSize();
    const data = try a.alloc(u8, 37 * page + 100);
    defer a.free(data);
    for (data, 0..) |*x, i| x.* = @truncate(i *% 29 +% (i >> 7));
    const f = try testFile(a, "warm", data);
    defer a.free(f.path);
    defer _ = std.c.unlink(f.path);
    defer _ = std.c.close(f.fd);
    try std.testing.expect(warmed(f.fd) == null);
    warm(f.fd); // not mapped: left alone
    map(f.fd);
    defer unmap(f.fd);
    try std.testing.expect(warmed(f.fd) == null);
    settle(f.fd);
    // Nothing taken as held, as after a load that pushed the file out: every row would be a pread.
    forget(f.fd);
    try std.testing.expect(!holds(f.fd, 5 * page, 256));
    warm(f.fd);
    warm(f.fd); // once a map
    while (warmed(f.fd).?.passes == 0) _ = usleep(1000);
    settle(f.fd);
    var row: [256]u8 = undefined;
    for ([_]usize{ 40, 9 * page - 100, 18 * page + 7, 27 * page + 1, 37 * page - 200 }) |off| {
        try std.testing.expect(holds(f.fd, off, 256));
        try std.testing.expect(readRow(f.fd, &row, off));
        try std.testing.expectEqualSlices(u8, data[off..][0..256], &row);
    }
    const c = counts(f.fd);
    try std.testing.expectEqual(@as(u64, 5), c.copies);
    try std.testing.expectEqual(@as(u64, 0), c.preads);
    // Pages taken back, as far as the map knows: the pread that finds one gone starts another pass, after which
    // every page is held again.
    forget(f.fd);
    try std.testing.expect(readRow(f.fd, &row, 3 * page + 5));
    try std.testing.expectEqual(@as(u64, 1), counts(f.fd).preads);
    while (warmed(f.fd).?.passes < 2) _ = usleep(1000);
    settle(f.fd);
    try std.testing.expect(holds(f.fd, 30 * page, 256));
    try std.testing.expectEqual(@as(u32, 0), warmed(f.fd).?.strikes);
}

test "engram_direct map: a pass reads the chunks whose pages are not held and no others, and a file unmapped mid-pass ends it" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const page = std.heap.pageSize();
    const data = try a.alloc(u8, 9 * page);
    defer a.free(data);
    @memset(data, 7);
    const f = try testFile(a, "pass", data);
    defer a.free(f.path);
    defer _ = std.c.unlink(f.path);
    defer _ = std.c.close(f.fd);
    map(f.fd);
    {
        defer unmap(f.fd);
        settle(f.fd);
        const m = mapOf(f.fd).?;
        // One page not held: the reader of its part reads that page's chunk alone.
        @atomicStore(u8, &m.held[4], 0, .monotonic);
        var w: Warm = .{};
        for (0..Warm.readers) |k| Warm.part(m.*, &w, k);
        try std.testing.expectEqual(@as(u64, Warm.chunk), w.read.load(.monotonic));
        try std.testing.expect(holds(f.fd, 4 * page, 16));
        // Told to stop, a reader reads nothing.
        forget(f.fd);
        w.stop.store(true, .release);
        for (0..Warm.readers) |k| Warm.part(m.*, &w, k);
        try std.testing.expectEqual(@as(u64, Warm.chunk), w.read.load(.monotonic));
        try std.testing.expect(!holds(f.fd, 0, 16));
    }
    // A map closed while its pass may still be running: the close waits for it.
    map(f.fd);
    warm(f.fd);
    unmap(f.fd);
    try std.testing.expect(!mapped(f.fd));
}

test "engram_direct map: a stretch of copies ends every stride and counts its own page-ins" {
    var c: Ctl = .{};
    c.page_ins.store(1000, .monotonic);
    // A stretch ends every `stride` copies.
    var ends: u32 = 0;
    for (0..3 * Ctl.stride) |_| ends += @intFromBool(c.strideEnds());
    try std.testing.expectEqual(@as(u32, 3), ends);
    // The page-ins of a stretch are its own: the count restarts at each end, across the counter's wrap too.
    try std.testing.expectEqual(@as(u32, 5), c.pagedIn(1005));
    try std.testing.expectEqual(@as(u32, 3), c.pagedIn(1008));
    c.page_ins.store(std.math.maxInt(u32) - 3, .monotonic);
    try std.testing.expectEqual(@as(u32, 14), c.pagedIn(10));
}

test "engram_direct map: under suspicion a row is copied only out of pages copied from before, and nothing known is thrown away" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const page = std.heap.pageSize();
    const data = try a.alloc(u8, 3 * page);
    defer a.free(data);
    for (data, 0..) |*x, i| x.* = @truncate(i *% 13 +% (i >> 7));
    const f = try testFile(a, "doubt", data);
    defer a.free(f.path);
    defer _ = std.c.unlink(f.path);
    defer _ = std.c.close(f.fd);
    map(f.fd);
    defer unmap(f.fd);
    // The kernel's answer, asked over more than one thread here (`ask_pages`), covers every page.
    settle(f.fd);
    try std.testing.expect(holds(f.fd, 0, data.len));
    const m = mapOf(f.fd).?;
    // A copy touches its row's page and no other.
    var row: [256]u8 = undefined;
    try std.testing.expect(m.copies(10, 256));
    try std.testing.expect(readRow(f.fd, &row, 10));
    try std.testing.expectEqualSlices(u8, data[10..][0..256], &row);
    try std.testing.expect(m.bits(10, 256) & Map.touched != 0);
    try std.testing.expect(m.bits(page + 10, 256) & Map.touched == 0);
    // The page-ins say `held` is stale, the kernel was asked a moment ago: suspicion stays until a pread's turn to
    // ask (`Ctl.miss`). The touched page's rows are still copies; an untouched page's row, held or not, is a pread,
    // which leaves it untouched; so is a row with one page of each kind.
    m.doubt(Ctl.stale_at);
    try std.testing.expectEqual(Trust.touched, m.ctl.?.trust.load(.monotonic));
    try std.testing.expect(m.copies(10, 256));
    try std.testing.expect(holds(f.fd, page + 10, 256) and !m.copies(page + 10, 256));
    try std.testing.expect(!m.copies(page - 100, 256));
    try std.testing.expect(readRow(f.fd, &row, page + 10));
    try std.testing.expectEqualSlices(u8, data[page + 10 ..][0..256], &row);
    try std.testing.expect(m.bits(page + 10, 256) & Map.touched == 0);
    try std.testing.expect(readRow(f.fd, &row, page - 100));
    try std.testing.expectEqualSlices(u8, data[page - 100 ..][0..256], &row);
    try std.testing.expect(m.bits(10, 256) & Map.touched != 0);
    // As many page-ins again in a stretch of touched pages' copies: a few are the copies that were in flight.
    m.doubt(Ctl.thrash_at - 1);
    try std.testing.expect(m.copies(10, 256));
    // Past `thrash_at`: the touched pages are being dropped too, and no row is a copy; rows read the same.
    m.doubt(Ctl.thrash_at);
    try std.testing.expectEqual(Trust.none, m.ctl.?.trust.load(.monotonic));
    try std.testing.expect(!m.copies(10, 256));
    try std.testing.expect(readRow(f.fd, &row, 10));
    try std.testing.expectEqualSlices(u8, data[10..][0..256], &row);
    try std.testing.expect(m.bits(10, 256) & Map.touched != 0);
    // An old question: the kernel is asked at once, and its answer ends the suspicion and keeps what was touched.
    m.ctl.?.asked_ns.store(0, .monotonic);
    m.doubt(Ctl.stale_at);
    settle(f.fd);
    // The file's counts: one row copied, three read by pread, the kernel asked at the map and now, stale four times.
    const c = counts(f.fd);
    try std.testing.expectEqual(@as(u64, 1), c.copies);
    try std.testing.expectEqual(@as(u64, 3), c.preads);
    try std.testing.expectEqual(@as(u32, 2), c.asked);
    try std.testing.expectEqual(@as(u32, 4), c.stale);
    try std.testing.expectEqual(Trust.held, m.ctl.?.trust.load(.monotonic));
    try std.testing.expect(holds(f.fd, 0, data.len));
    try std.testing.expect(m.bits(10, 256) & Map.touched != 0);
    try std.testing.expect(m.copies(page + 10, 256));
}

test "engram_direct map: the kernel's answer decides what is held, and a page it no longer holds is no longer touched" {
    // Held and touched, still held: both kept. mincore's other bits (referenced, modified) are not `touched`.
    try std.testing.expectEqual(Map.in_cache | Map.touched, merged(Map.in_cache | Map.touched, 1));
    try std.testing.expectEqual(Map.in_cache, merged(Map.in_cache, 0x1f));
    try std.testing.expectEqual(Map.in_cache, merged(0, 0x03));
    // Dropped by the kernel: neither, whatever it was.
    try std.testing.expectEqual(@as(u8, 0), merged(Map.in_cache | Map.touched, 0));
    try std.testing.expectEqual(@as(u8, 0), merged(Map.in_cache, 0x1e));
}
