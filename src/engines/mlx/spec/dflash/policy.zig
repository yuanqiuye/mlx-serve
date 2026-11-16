//! DFlash round policy for a sparse target, where a verify row is NOT free (GLM-5.3-Flash: every
//! extra row routes to more distinct experts). Pure data, no MLX: the generator feeds it each
//! round and reads the next decision back.
//!
//! A round drafts once, then verifies the first R rows (t1 and R - 1 drafts). R maximizes
//! emitted tokens per ms under (a) a per-position acceptance probability calibrated online
//! from the drafter's confidence and (b) the cost of an R-row verify. R = 1 is a plain round.
//! When drafts keep failing to pay for their own forward, drafting is skipped with a probe backoff.
const std = @import("std");

/// t1 plus up to 15 drafts.
pub const MAX_ROWS: u32 = 16;
pub const MAX_DRAFTS: u32 = MAX_ROWS - 1;

/// Round ms by verify rows; index = rows, 0 unused.
pub const Costs = [MAX_ROWS + 1]f32;

/// GLM-5.3-Flash verify ladder on an M5 Ultra at 1k context (ms by rows, linear between).
const LADDER = [_][2]f32{ .{ 1, 17.7 }, .{ 2, 25.5 }, .{ 3, 29.6 }, .{ 4, 33.6 }, .{ 5, 36.8 }, .{ 6, 42.4 }, .{ 7, 44.8 }, .{ 8, 47.9 }, .{ 9, 51.1 }, .{ 12, 60.5 }, .{ 16, 69.1 } };

/// The ladder scaled so one row costs `plain_ms`: a faster or slower machine moves it whole.
pub fn priorCosts(plain_ms: f32) Costs {
    var out: Costs = undefined;
    out[0] = 0;
    const scale = plain_ms / LADDER[0][1];
    for (1..MAX_ROWS + 1) |r| {
        const rows: f32 = @floatFromInt(r);
        var ms: f32 = LADDER[LADDER.len - 1][1];
        for (LADDER[0 .. LADDER.len - 1], LADDER[1..]) |lo, hi| {
            if (rows <= hi[0]) {
                ms = lo[1] + (hi[1] - lo[1]) * (rows - lo[0]) / (hi[0] - lo[0]);
                break;
            }
        }
        out[r] = ms * scale;
    }
    return out;
}

/// A plain round's ms when nothing has been measured yet.
pub const PRIOR_PLAIN_MS: f32 = LADDER[0][1];

pub const Choice = struct {
    /// Rows to verify, t1 included (1 = a plain round).
    rows: u32,
    /// Expected tokens the round emits.
    tokens: f32,
    /// Verify cost of `rows`.
    ms: f32,
};

/// A row count within this fraction of the best rate loses to the cheaper one: the model is not
/// that exact, and a verify's cost is certain where its tokens are not.
const ROW_TIE: f32 = 0.02;

/// The row count with the best expected tokens per ms. `p[k]` is the probability that draft k is
/// accepted given drafts 0..k-1 were; `costs` are the verify ms by rows.
pub fn chooseRows(p: []const f32, costs: *const Costs) Choice {
    return chooseRowsAt(p, costs, 0);
}

/// `chooseRows` for a round that costs `fixed_ms` whatever it verifies (the draft's forward): the row count with the
/// best expected tokens per ms of the whole round. A round's rate is its tokens over its draft and its verify, so a
/// row that adds less than the verify's own rate can still raise the round's.
pub fn chooseRowsAt(p: []const f32, costs: *const Costs, fixed_ms: f32) Choice {
    const n: usize = @min(p.len, MAX_DRAFTS); // typed: a bare @min narrows to u4 and n + 2 overflows at 15
    const fixed: f32 = if (std.math.isFinite(fixed_ms) and fixed_ms > 0) fixed_ms else 0;
    var tokens: [MAX_ROWS + 1]f32 = undefined;
    tokens[1] = 1;
    var alive: f32 = 1;
    for (p[0..n], 0..) |pk, k| {
        alive *= if (pk >= 0) @min(pk, 1) else 0; // a NaN is no acceptance
        tokens[k + 2] = tokens[k + 1] + alive;
    }
    var best: f32 = 0;
    for (1..n + 2) |r| best = @max(best, tokens[r] / (fixed + costs[r]));
    for (1..n + 2) |r| {
        if (tokens[r] / (fixed + costs[r]) >= best * (1 - ROW_TIE)) return .{ .rows = @intCast(r), .tokens = tokens[r], .ms = costs[r] };
    }
    return .{ .rows = 1, .tokens = 1, .ms = costs[1] };
}

/// The row count worth the most at `rate` tokens per ms: the expected tokens a verify of r rows emits less what its ms
/// would have bought at the rate the request is running at. Unlike the best ratio of one round, it prices a row by
/// what the round's time is worth across rounds: a sure round takes every row, a doubtful one stops where a row's
/// share of a token falls under `rate` times its ms. The fewer rows on a tie.
pub fn chooseRowsValue(p: []const f32, costs: *const Costs, rate: f32) Choice {
    const n: usize = @min(p.len, MAX_DRAFTS);
    const per_ms: f32 = if (std.math.isFinite(rate) and rate > 0) rate else 0;
    var best: Choice = .{ .rows = 1, .tokens = 1, .ms = costs[1] };
    var best_value: f32 = 1 - per_ms * costs[1];
    var tokens: f32 = 1;
    var alive: f32 = 1;
    for (p[0..n], 0..) |pk, k| {
        alive *= if (pk >= 0) @min(pk, 1) else 0; // a NaN is no acceptance
        tokens += alive;
        const value = tokens - per_ms * costs[k + 2];
        if (value > best_value) {
            best_value = value;
            best = .{ .rows = @intCast(k + 2), .tokens = tokens, .ms = costs[k + 2] };
        }
    }
    return best;
}

/// The rate a request is running at: tokens per ms over its recent rounds and plain ticks, the plain step's until
/// a round has run.
pub const Rate = struct {
    const ALPHA: f32 = 0.05;

    tokens: f32 = 0,
    ms: f32 = 0,

    pub fn perMs(self: *const Rate, plain_ms: f32) f32 {
        return if (self.ms > 0) self.tokens / self.ms else 1 / plain_ms;
    }

    /// A round or tick emitted `tokens` tokens in `ms`.
    pub fn record(self: *Rate, tokens: f32, ms: f32, plain_ms: f32) void {
        if (!std.math.isFinite(ms) or ms <= 0 or !(tokens >= 0)) return;
        if (self.ms <= 0) {
            self.tokens = 1;
            self.ms = plain_ms;
        }
        // A stall (a kernel's first compile, a foreign GPU job) is not the request's rate.
        const took = @min(ms, 4 * self.ms);
        self.tokens += ALPHA * (tokens - self.tokens);
        self.ms += ALPHA * (took - self.ms);
    }
};

/// What a verify of r rows costs on this model and this machine, and what a draft costs, learned from the rounds
/// themselves the way `PlainCost` learns the plain step. The prior ladder is another model's on another build: its
/// second row costs 7.8 ms where DeepSeek-V4.1's fused mxfp4 kernels take about 3.5, so a round priced off it leaves
/// drafts unverified that would have paid for their row.
pub const VerifyCost = struct {
    /// A row count's first samples keep the fastest: a stall (a kernel's first compile) only reads slower.
    const SEED: u32 = 3;
    /// Samples before a row count's own time prices it.
    const TRUST: u32 = 2;

    /// Verify ms by rows (index = rows, 0 unused), each over `seen` samples.
    ms: [MAX_ROWS + 1]f32 = @splat(0),
    seen: [MAX_ROWS + 1]u32 = @splat(0),
    /// A draft's forward, the cost a drafted round pays whatever it verifies.
    draft_ms: f32 = 0,
    draft_seen: u32 = 0,

    fn learn(cur: *f32, seen: *u32, took: f32) void {
        if (!std.math.isFinite(took) or took <= 0) return;
        if (seen.* < SEED) {
            cur.* = if (seen.* == 0) took else @min(cur.*, took);
        } else cur.* += 0.1 * (std.math.clamp(took, cur.* * 0.8, cur.* * 1.25) - cur.*);
        seen.* +|= 1;
    }

    /// A verify of `rows` rows (t1 included) took `took` ms.
    pub fn sample(self: *VerifyCost, rows: usize, took: f32) void {
        if (rows == 0 or rows > MAX_ROWS) return;
        learn(&self.ms[rows], &self.seen[rows], took);
    }

    /// A draft's forward took `took` ms.
    pub fn sampleDraft(self: *VerifyCost, took: f32) void {
        learn(&self.draft_ms, &self.draft_seen, took);
    }

    /// What a round's draft costs: 0 until one is measured.
    pub fn draftMs(self: *const VerifyCost) f32 {
        return if (self.draft_seen >= TRUST) self.draft_ms else 0;
    }

    fn known(self: *const VerifyCost, r: usize) bool {
        return self.seen[r] >= TRUST;
    }

    /// The verify's ms by rows: a measured row count its own time; one between two measured ones on their line; one
    /// past the last measured one that row's time plus the prior ladder's steps (scaled to `plain_ms`); never cheaper
    /// for more rows. With nothing measured this is `priorCosts(plain_ms)`.
    pub fn costs(self: *const VerifyCost, plain_ms: f32) Costs {
        const prior = priorCosts(plain_ms);
        var out = prior;
        var last: usize = 0; // the last measured row count at or below r
        for (1..MAX_ROWS + 1) |r| {
            if (self.known(r)) {
                out[r] = self.ms[r];
                last = r;
                continue;
            }
            if (last == 0) continue; // below the first measured row: the prior
            const next: usize = for (r + 1..MAX_ROWS + 1) |q| {
                if (self.known(q)) break q;
            } else 0;
            if (next != 0) {
                const t = @as(f32, @floatFromInt(r - last)) / @as(f32, @floatFromInt(next - last));
                out[r] = self.ms[last] + (self.ms[next] - self.ms[last]) * t;
            } else out[r] = out[r - 1] + (prior[r] - prior[r - 1]);
        }
        for (2..MAX_ROWS + 1) |r| out[r] = @max(out[r], out[r - 1]);
        return out;
    }

    /// `1:17.6,2:21.4,...`: the measured row counts, for `[spec-stats]`.
    pub fn format(self: *const VerifyCost, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        var first = true;
        for (1..MAX_ROWS + 1) |r| {
            if (!self.known(r)) continue;
            if (!first) w.writeAll(",") catch break;
            first = false;
            w.print("{d}:{d:.1}", .{ r, self.ms[r] }) catch break;
        }
        return w.buffered();
    }
};

/// Probability that a draft is accepted given its predecessors were, by the drafter's confidence in
/// it: online bins over `-log10(1 - conf)` (the selector's shares crowd 0.9..1, where a linear axis has one
/// bin), each pulled toward a logistic prior fitted on recorded rounds by a few pseudo-counts and forgetting
/// its old rounds, so a change of content class re-calibrates in a few rounds.
pub const Calibration = struct {
    const BINS = 12;
    const PER_UNIT: f32 = 4;
    /// Pseudo-observations the prior is worth.
    const PRIOR_WEIGHT: f32 = 3;
    /// A bin forgets this share of its history per observation it takes.
    const KEEP: f32 = 0.97;

    hit: [BINS]f32 = @splat(0),
    seen: [BINS]f32 = @splat(0),
    /// The drafter's confidence is its own acceptance estimate (DSpark's confidence head), the prior the bins start
    /// from; else the logistic fit.
    head: bool = false,
    /// Pseudo-observations the prior is worth here. A trained confidence head (`HEAD_WEIGHT`) is worth more than
    /// the fitted curve: over 2,993 DeepSeek-V4.1 rounds its bins sat within a few points of the acceptance they
    /// named, and bins pulled by three pseudo-counts chased each stretch's noise (a replay: 78.6 tok/s against 79.0
    /// at twelve, 77.9 against 75.6 for the head alone when the head read 1.5x too sure).
    prior_weight: f32 = PRIOR_WEIGHT,

    pub const HEAD_WEIGHT: f32 = 12;

    /// 0 at a coin flip's confidence, 1 at 0.9, 2 at 0.99, capped at 3.
    fn axis(conf: f32) f32 {
        return -std.math.log10(@max(1 - std.math.clamp(conf, 0, 1), 1e-3));
    }

    fn bin(conf: f32) usize {
        return @min(BINS - 1, @as(usize, @intFromFloat(axis(conf) * PER_UNIT)));
    }

    fn prior(self: *const Calibration, conf: f32) f32 {
        if (self.head) return std.math.clamp(conf, 0, 1);
        return 1 / (1 + @exp(-(-1.15 + 1.58 * axis(conf))));
    }

    pub fn probability(self: *const Calibration, conf: f32) f32 {
        const b = bin(conf);
        return (self.hit[b] + self.prior_weight * self.prior(conf)) / (self.seen[b] + self.prior_weight);
    }

    pub fn observe(self: *Calibration, conf: f32, accepted: bool) void {
        const b = bin(conf);
        self.hit[b] = self.hit[b] * KEEP + @as(f32, if (accepted) 1 else 0);
        self.seen[b] = self.seen[b] * KEEP + 1;
    }
};

/// What a plain round costs, learned from the plain rounds themselves and nothing else: slowly, and by at
/// most a quarter per sample, so a stall or a foreign GPU job cannot pose as the machine's speed. With
/// `seed_samples` the first samples replace the prior (another model on another machine) whatever they read,
/// keeping the fastest, since a stall only reads slower.
pub const PlainCost = struct {
    /// DSpark's: learned from the first three plain rounds on this machine.
    pub const seeded: PlainCost = .{ .seed_samples = 3 };

    ms: f32 = PRIOR_PLAIN_MS,
    seed_samples: u32 = 0,
    measured: u32 = 0,

    pub fn sample(self: *PlainCost, ms: f32) void {
        if (!std.math.isFinite(ms) or ms <= 0) return;
        if (self.measured < self.seed_samples) return self.seed(ms);
        const bounded = std.math.clamp(ms, self.ms * 0.8, self.ms * 1.25);
        self.ms += 0.1 * (bounded - self.ms);
    }

    fn seed(self: *PlainCost, ms: f32) void {
        self.ms = if (self.measured == 0) ms else @min(self.ms, ms);
        self.measured += 1;
    }
};

/// How often a copied continuation survives verify, by how far back it agreed with the context (under or
/// over `STRONG`): the share of positions reached that were kept, forgetting old rounds.
pub const CopyStats = struct {
    pub const STRONG: u32 = 16;
    const PRIOR = [2]f32{ 0.7, 0.9 };
    const PRIOR_WEIGHT: f32 = 3;
    const KEEP: f32 = 0.95;

    hit: [2]f32 = .{ PRIOR[0] * PRIOR_WEIGHT, PRIOR[1] * PRIOR_WEIGHT },
    tries: [2]f32 = .{ PRIOR_WEIGHT, PRIOR_WEIGHT },

    fn class(suffix: u32) usize {
        return @intFromBool(suffix >= STRONG);
    }

    pub fn probability(self: *const CopyStats, suffix: u32) f32 {
        const c = class(suffix);
        return self.hit[c] / self.tries[c];
    }

    /// A round that verified `drafts` copied tokens and kept `accepted` of them.
    pub fn observe(self: *CopyStats, suffix: u32, drafts: u32, accepted: u32) void {
        const c = class(suffix);
        self.hit[c] = self.hit[c] * KEEP + @as(f32, @floatFromInt(accepted));
        self.tries[c] = self.tries[c] * KEEP + @as(f32, @floatFromInt(@min(accepted + 1, drafts)));
    }

    /// Verify rows (t1 included) for a copy of up to `avail` drafts: each priced at this match's survival
    /// against the verify ladder scaled to `plain_ms`.
    pub fn rowsFor(self: *const CopyStats, suffix: u32, avail: usize, plain_ms: f32) u32 {
        const costs = priorCosts(plain_ms);
        return self.rowsAt(suffix, avail, &costs);
    }

    /// `rowsFor` against a verify ladder of the caller's (a model's own, `VerifyCost.costs`).
    pub fn rowsAt(self: *const CopyStats, suffix: u32, avail: usize, costs: *const Costs) u32 {
        var p: [MAX_DRAFTS]f32 = undefined;
        const n = @min(avail, p.len);
        @memset(p[0..n], self.probability(suffix));
        return chooseRows(p[0..n], costs).rows;
    }
};

/// Whether this round drafts at all. Drafting costs a forward whether or not its rows are kept, so
/// a stretch where it does not pay is skipped with a probe backoff: each probe that loses doubles
/// the plain rounds before the next, a probe that wins clears the debt.
pub const Drafting = struct {
    /// Longest stretch of plain rounds between two probes.
    const MAX_PERIOD: u32 = 32;
    const ALPHA: f32 = 0.3;

    /// Averages over drafted rounds of the plain-round ms their tokens replaced and of the ms they took:
    /// the ratio weighs a round by its time, where a mean of per-round ratios swings on every short miss.
    gain: f32 = 0,
    wall: f32 = 0,
    /// Plain rounds still owed before the next draft.
    owed: u32 = 0,
    /// Plain rounds the next losing round will owe.
    period: u32 = 1,
    /// A round loses when its gain is under this share of its wall. Below 1 it keeps drafting through rounds a little
    /// under break-even: skipping a stretch forgoes its good rounds too, which costs more than the drafts saved.
    margin: f32 = 1,
    /// Rounds skipped, for stats.
    skipped: u64 = 0,
    /// Drafted rounds that followed a skip stretch.
    probes: u64 = 0,
    after_skip: bool = false,

    pub fn shouldDraft(self: *Drafting) bool {
        if (self.owed > 0) {
            self.owed -= 1;
            self.skipped += 1;
            self.after_skip = true;
            return false;
        }
        if (self.after_skip) self.probes += 1;
        self.after_skip = false;
        return true;
    }

    /// A round counts for at most this many times the rounds before it (the first: its own gain).
    const STALL: f32 = 2;

    /// A drafted round: `gain` = the ms of plain rounds its tokens would have cost, `took` = what it took. A stall (a
    /// kernel's first compile, a foreign GPU job) is not what a round costs, as for `PlainCost`.
    pub fn record(self: *Drafting, gain: f32, took: f32) void {
        const wall = @min(took, STALL * if (self.wall > 0) self.wall else gain);
        if (self.wall <= 0) {
            self.gain = gain;
            self.wall = wall;
        } else {
            self.gain += ALPHA * (gain - self.gain);
            self.wall += ALPHA * (wall - self.wall);
        }
        if (self.gain >= self.wall * self.margin) {
            self.owed = 0;
            self.period = @max(1, self.period / 2);
        } else {
            self.owed = self.period;
            self.period = @min(self.period * 2, MAX_PERIOD);
        }
    }
};

/// Rows one DSpark group round verifies in all: the module batches its compiled stretches up to this many.
pub const MAX_GROUP_ROWS: u32 = 16;

pub const GroupCosts = [MAX_GROUP_ROWS + 1]f32;

/// Verify rows for slots sharing one batched verify. `p[k]` is slot k's per-draft acceptance as `chooseRows` reads it,
/// `costs[r]` the verify's ms at `r` rows in all, the slots' `t1` rows included. The most tokens any `r` rows buy are the
/// `r` likeliest drafts (a slot's in order), so the choice is the `r` with the best expected tokens per ms, the fewer
/// rows on a near tie. Writes slot k's drafts to verify to `drafts[k]`; returns the expected tokens.
pub fn chooseGroupRows(p: []const []const f32, costs: *const GroupCosts, drafts: []u32) f32 {
    const n = p.len;
    std.debug.assert(n >= 1 and n <= MAX_GROUP_ROWS);
    var alive: [MAX_GROUP_ROWS]f32 = @splat(1);
    var next: [MAX_GROUP_ROWS]u32 = @splat(0);
    var picks: [MAX_GROUP_ROWS]u8 = undefined;
    var tokens: [MAX_GROUP_ROWS + 1]f32 = undefined;
    tokens[n] = @floatFromInt(n);
    var r = n;
    while (r < MAX_GROUP_ROWS) {
        var pick: ?usize = null;
        var gain: f32 = 0;
        for (p, 0..) |pk, k| {
            if (next[k] >= pk.len) continue;
            const g = alive[k] * (if (pk[next[k]] >= 0) @min(pk[next[k]], 1) else 0); // a NaN is no acceptance
            if (g > gain) {
                pick = k;
                gain = g;
            }
        }
        const k = pick orelse break;
        alive[k] = gain;
        next[k] += 1;
        picks[r - n] = @intCast(k);
        r += 1;
        tokens[r] = tokens[r - 1] + gain;
    }
    var best: f32 = 0;
    for (n..r + 1) |rr| best = @max(best, tokens[rr] / costs[rr]);
    var chosen = n;
    for (n..r + 1) |rr| if (tokens[rr] / costs[rr] >= best * (1 - ROW_TIE)) {
        chosen = rr;
        break;
    };
    @memset(drafts[0..n], 0);
    for (picks[0 .. chosen - n]) |k| drafts[k] += 1;
    return tokens[chosen];
}

/// What a DSpark group round's verify costs: `base + per_state * states + per_row * (rows - states)` ms, one row a slot
/// being the group's plain step. Fitted on an M5 Ultra (`DSV41_GROUP`), scaled by what this machine's rounds took: slowly,
/// and by at most a quarter per sample (`PlainCost`'s rule).
pub const GroupCost = struct {
    const BASE: f32 = 12;
    const PER_STATE: f32 = 4.6;
    const PER_ROW: f32 = 4.4;

    scale: f32 = 1,

    fn prior(states: usize, rows: usize) f32 {
        return BASE + PER_STATE * @as(f32, @floatFromInt(states)) + PER_ROW * @as(f32, @floatFromInt(rows -| states));
    }

    pub fn ms(self: *const GroupCost, states: usize, rows: usize) f32 {
        return self.scale * prior(states, rows);
    }

    /// What one more verify row costs.
    pub fn rowMs(self: *const GroupCost) f32 {
        return self.scale * PER_ROW;
    }

    /// The verify's ms by rows for a group of `states` slots (index = rows, below `states` unused).
    pub fn costs(self: *const GroupCost, states: usize) GroupCosts {
        var out: GroupCosts = @splat(0);
        for (states..MAX_GROUP_ROWS + 1) |r| out[r] = self.ms(states, r);
        return out;
    }

    /// A verify of `rows` rows over `states` slots took `took` ms.
    pub fn sample(self: *GroupCost, states: usize, rows: usize, took: f32) void {
        if (!std.math.isFinite(took) or took <= 0) return;
        const want = took / prior(states, rows);
        self.scale += 0.1 * (std.math.clamp(want, self.scale * 0.8, self.scale * 1.25) - self.scale);
    }
};

/// Per-draft-position acceptance and the verify widths a request ran, for `[spec-stats]`.
pub const PositionStats = struct {
    reached: [MAX_DRAFTS]u64 = @splat(0),
    accepted: [MAX_DRAFTS]u64 = @splat(0),
    /// Rounds by verify rows; index = rows.
    rows: [MAX_ROWS + 1]u64 = @splat(0),

    /// A round that verified `drafts` drafts and kept `accepted` of them: it reached every position
    /// up to the first rejection and no further.
    pub fn record(self: *PositionStats, drafts: u32, accepted: u32) void {
        const m: usize = @min(drafts, MAX_DRAFTS);
        self.rows[m + 1] += 1;
        for (0..m) |k| {
            if (k > accepted) break;
            self.reached[k] += 1;
            if (k < accepted) self.accepted[k] += 1;
        }
    }

    /// `1.00/0.95/...`: conditional acceptance by draft position, down to the deepest one any round reached.
    pub fn formatAcc(self: *const PositionStats, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        for (self.reached, self.accepted, 0..) |reached, kept, k| {
            if (reached == 0) break;
            if (k > 0) w.writeAll("/") catch break;
            w.print("{d:.2}", .{@as(f32, @floatFromInt(kept)) / @as(f32, @floatFromInt(reached))}) catch break;
        }
        return w.buffered();
    }

    /// `1:12,2:3,8:100`: rounds by verify rows.
    pub fn formatRows(self: *const PositionStats, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        var first = true;
        for (self.rows, 0..) |n, rows| {
            if (n == 0) continue;
            if (!first) w.writeAll(",") catch break;
            first = false;
            w.print("{d}:{d}", .{ rows, n }) catch break;
        }
        return w.buffered();
    }
};

const testing = std.testing;

fn approx(a: f32, b: f32) !void {
    try testing.expectApproxEqAbs(b, a, 0.02);
}

test "priorCosts: scaled to the measured plain step, never cheaper for more rows" {
    const c = priorCosts(17.7);
    try testing.expectApproxEqAbs(@as(f32, 17.7), c[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 25.5), c[2], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 47.9), c[8], 1e-3);
    var r: u32 = 2;
    while (r <= MAX_ROWS) : (r += 1) try testing.expect(c[r] >= c[r - 1]);
    // A faster machine's plain step moves the whole ladder with it.
    const half = priorCosts(8.85);
    try testing.expectApproxEqAbs(c[8] / 2, half[8], 1e-3);
}

test "chooseRowsAt: a draft already paid for buys rows the verify's own rate would not" {
    // One draft at 40%: the prior ladder's second row (+7.8 ms on 17.7) does not pay by the verify's rate alone...
    const costs = priorCosts(17.7);
    try testing.expectEqual(@as(u32, 1), chooseRows(&[_]f32{0.40}, &costs).rows);
    try testing.expectEqual(@as(u32, 1), chooseRowsAt(&[_]f32{0.40}, &costs, 0).rows);
    // ...and does once the round's 6 ms draft is in its cost: 1.4 tokens in 31.5 ms beat 1 in 23.7.
    try testing.expectEqual(@as(u32, 2), chooseRowsAt(&[_]f32{0.40}, &costs, 6).rows);
    // A draft that almost never lands is still left alone, and a poisoned fixed cost counts as none.
    try testing.expectEqual(@as(u32, 1), chooseRowsAt(&[_]f32{0.05}, &costs, 6).rows);
    try testing.expectEqual(chooseRows(&[_]f32{ 0.9, 0.9 }, &costs).rows, chooseRowsAt(&[_]f32{ 0.9, 0.9 }, &costs, std.math.nan(f32)).rows);
}

test "chooseRowsValue: rows while a draft's share of a token beats the rate times its ms" {
    var v: VerifyCost = .{};
    for (1..7) |r| for (0..3) |_| v.sample(r, 17.5 + 3.5 * @as(f32, @floatFromInt(r - 1)));
    const costs = v.costs(17.5);
    // At 0.07 tokens a ms a 3.5 ms row must carry a quarter of a token.
    try testing.expectEqual(@as(u32, 6), chooseRowsValue(&[_]f32{ 0.95, 0.95, 0.95, 0.95, 0.95 }, &costs, 0.07).rows);
    try testing.expectEqual(@as(u32, 1), chooseRowsValue(&[_]f32{ 0.2, 0.9, 0.9, 0.9, 0.9 }, &costs, 0.07).rows);
    try testing.expectEqual(@as(u32, 2), chooseRowsValue(&[_]f32{ 0.3, 0.5, 0.9, 0.9, 0.9 }, &costs, 0.07).rows);
    // 0.6 then 0.5: the third row carries 0.30 of a token, the fourth 0.15.
    const c = chooseRowsValue(&[_]f32{ 0.6, 0.5, 0.5, 0.5, 0.5 }, &costs, 0.07);
    try testing.expectEqual(@as(u32, 3), c.rows);
    try approx(c.tokens, 1.9);
    try approx(c.ms, 24.5);
    // A slower request buys more rows with the same drafts, a faster one fewer.
    try testing.expect(chooseRowsValue(&[_]f32{ 0.6, 0.5, 0.5, 0.5, 0.5 }, &costs, 0.03).rows > 3);
    try testing.expect(chooseRowsValue(&[_]f32{ 0.6, 0.5, 0.5, 0.5, 0.5 }, &costs, 0.12).rows < 3);
    // A dip in the ladder is taken through: rows 5 and 6 together pay where row 5 alone does not.
    var dip = costs;
    dip[5] = dip[4] + 8;
    dip[6] = dip[5] + 0.5;
    try testing.expectEqual(@as(u32, 6), chooseRowsValue(&[_]f32{ 1, 1, 1, 0.4, 1 }, &dip, 0.07).rows);
    // No drafts, a poisoned rate or probability: still an answer.
    try testing.expectEqual(@as(u32, 1), chooseRowsValue(&.{}, &costs, 0.07).rows);
    try testing.expectEqual(@as(u32, 6), chooseRowsValue(&[_]f32{ 1, 1, 1, 1, 1 }, &costs, std.math.nan(f32)).rows);
    try testing.expectEqual(@as(u32, 2), chooseRowsValue(&[_]f32{ 1, std.math.nan(f32), 1 }, &costs, 0.01).rows);
}

test "Rate: the plain step's until a round runs, then the recent rounds', a stall bounded" {
    var r: Rate = .{};
    try approx(r.perMs(17.5) * 17.5, 1);
    for (0..200) |_| r.record(3, 30, 17.5);
    try approx(r.perMs(17.5) * 10, 1);
    const before = r.perMs(17.5);
    r.record(1, 5000, 17.5);
    try testing.expect(r.perMs(17.5) > before * 0.8);
    r.record(1, std.math.nan(f32), 17.5);
    r.record(1, 0, 17.5);
    try testing.expect(r.perMs(17.5) > before * 0.8);
}

test "Calibration: a head's weight holds its bins nearer the head than the fitted curve's three counts" {
    var light = Calibration{ .head = true };
    var heavy = Calibration{ .head = true, .prior_weight = Calibration.HEAD_WEIGHT };
    try approx(light.probability(0.7), 0.7);
    try approx(heavy.probability(0.7), 0.7);
    for (0..3) |_| {
        light.observe(0.7, false);
        heavy.observe(0.7, false);
    }
    try testing.expect(heavy.probability(0.7) > light.probability(0.7));
    try testing.expect(heavy.probability(0.7) < 0.7);
}

test "VerifyCost: the prior until rows are measured, then each row count's own time" {
    var v: VerifyCost = .{};
    const prior = priorCosts(17.5);
    for (v.costs(17.5), prior) |got, want| try approx(got, want);
    try testing.expectEqual(@as(f32, 0), v.draftMs());
    // One sample is not trusted (a first run compiles its kernel); the fastest of the first samples is.
    v.sample(2, 90);
    try approx(v.costs(17.5)[2], prior[2]);
    v.sample(2, 21);
    v.sample(2, 22);
    try approx(v.costs(17.5)[2], 21);
    // Rows 1 and 4 measured: 3 sits on the line from 2 to 4, 5 and 6 step from 4 by the prior's steps.
    for (0..3) |_| v.sample(1, 17.5);
    for (0..3) |_| v.sample(4, 28);
    const c = v.costs(17.5);
    try approx(c[1], 17.5);
    try approx(c[3], 24.5);
    try approx(c[4], 28);
    try approx(c[5], 28 + (prior[5] - prior[4]));
    try approx(c[6], c[5] + (prior[6] - prior[5]));
    for (2..MAX_ROWS + 1) |r| try testing.expect(c[r] >= c[r - 1]);
    // Later samples move a row by a tenth of a bounded step: a stall cannot pose as the cost.
    v.sample(2, 500);
    try approx(v.costs(17.5)[2], 21 + 0.1 * (21 * 1.25 - 21));
    // A slower measured row never prices a wider verify below a narrower one.
    for (0..3) |_| v.sample(3, 40);
    const d = v.costs(17.5);
    try testing.expect(d[4] >= d[3]);
    // The draft: trusted from its second sample; bad samples and row counts are ignored.
    v.sampleDraft(5);
    v.sampleDraft(4);
    try approx(v.draftMs(), 4);
    v.sampleDraft(-1);
    v.sample(0, 3);
    v.sample(MAX_ROWS + 1, 3);
    v.sample(2, std.math.nan(f32));
    try approx(v.draftMs(), 4);
    var buf: [128]u8 = undefined;
    try testing.expect(std.mem.startsWith(u8, v.format(&buf), "1:17.5,2:"));
}

test "VerifyCost: this model's ladder verifies a draft the prior ladder leaves" {
    // DeepSeek-V4.1 on the fused mxfp4 kernels: 17.5 ms plain, about 3.5 ms a further row, a 6 ms draft.
    var v: VerifyCost = .{};
    for (1..7) |r| for (0..3) |_| v.sample(r, 17.5 + 3.5 * @as(f32, @floatFromInt(r - 1)));
    for (0..3) |_| v.sampleDraft(6);
    const own = v.costs(17.5);
    const prior = priorCosts(17.5);
    const p = [_]f32{ 0.35, 0.6, 0.6, 0.6, 0.6 };
    try testing.expectEqual(@as(u32, 1), chooseRows(&p, &prior).rows);
    try testing.expect(chooseRowsAt(&p, &own, v.draftMs()).rows >= 3);
}

test "chooseRows: a plain round when nothing pays, the whole chain when everything does" {
    const costs = priorCosts(17.7);
    const weak = [_]f32{ 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1 };
    try testing.expectEqual(@as(u32, 1), chooseRows(&weak, &costs).rows);
    const sure: [7]f32 = @splat(0.96);
    const c = chooseRows(&sure, &costs);
    try testing.expectEqual(@as(u32, 8), c.rows);
    try approx(c.tokens, 6.96);
    try approx(c.ms, 47.9);
}

test "chooseRows: a fifteen-draft chain is priced through its last row, and a poisoned input still answers" {
    const costs = priorCosts(17.7);
    const sure: [15]f32 = @splat(0.98);
    const c = chooseRows(&sure, &costs);
    try testing.expect(c.rows >= 8 and c.rows <= 16);
    // A NaN probability must not leave the function without a choice.
    const bad: [3]f32 = .{ std.math.nan(f32), 0.9, 0.9 };
    try testing.expect(chooseRows(&bad, &costs).rows >= 1);
}

test "chooseRows: the second row pays for its own +8 ms jump only past 48% acceptance" {
    const costs = priorCosts(17.7);
    try testing.expectEqual(@as(u32, 1), chooseRows(&[_]f32{0.40}, &costs).rows);
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{0.60}, &costs).rows);
}

test "chooseRows: rows past a poor draft are not bought for the confident draft behind it" {
    const costs = priorCosts(17.7);
    // Draft 1 almost never lands, so draft 2 is only reached 5% of the time.
    const c = chooseRows(&[_]f32{ 0.95, 0.05, 0.99, 0.99 }, &costs);
    try testing.expectEqual(@as(u32, 2), c.rows);
}

test "chooseRows: within 2% of the best rate the cheaper verify wins" {
    var costs: Costs = @splat(20.0);
    costs[1] = 10.0;
    costs[2] = 12.0;
    costs[3] = 12.2;
    // The third row buys 1.5% over two rows: not worth verifying.
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{ 0.5, 0.0958 }, &costs).rows);
    // At 15% it is.
    try testing.expectEqual(@as(u32, 3), chooseRows(&[_]f32{ 0.5, 0.5 }, &costs).rows);
}

test "Calibration: no data answers the prior, then the data takes over, bin by bin" {
    var c = Calibration{};
    try approx(c.probability(0.9), 0.606);
    try approx(c.probability(0.2), 0.269);
    for (0..40) |_| c.observe(0.9, false);
    try testing.expect(c.probability(0.9) < 0.3);
    try approx(c.probability(0.2), 0.269);
    // The bin is a confidence range: 0.88 shares 0.9's evidence.
    try testing.expect(c.probability(0.88) < 0.3);
}

test "Calibration: an old regime fades" {
    var c = Calibration{};
    for (0..80) |_| c.observe(0.7, true);
    try testing.expect(c.probability(0.7) > 0.9);
    for (0..80) |_| c.observe(0.7, false);
    try testing.expect(c.probability(0.7) < 0.2);
}

// A drafted round is judged by `record(gain, wall)`: the plain-round ms its tokens would have cost, against
// the ms it took. Ten against ten is break-even.

test "PlainCost: a stall moves it little, a steady new speed is reached" {
    var c = PlainCost{};
    c.sample(400); // a compile stall, a foreign GPU job
    try testing.expect(c.ms > PRIOR_PLAIN_MS and c.ms < 18.5);
    var faster = PlainCost{};
    for (0..80) |_| faster.sample(14.0);
    try testing.expect(faster.ms < 14.5 and faster.ms >= 14.0);
    var slower = PlainCost{};
    for (0..80) |_| slower.sample(22.0);
    try testing.expect(slower.ms > 21.0 and slower.ms <= 22.0);
}

test "PlainCost: unseeded (DFlash) keeps the prior and walks from it, whatever the first samples read" {
    var c = PlainCost{};
    for (0..3) |_| c.sample(13.0);
    try testing.expect(c.ms > 16.5 and c.ms < PRIOR_PLAIN_MS);
    var slow = PlainCost{};
    for (0..3) |_| slow.sample(120.0);
    try testing.expect(slow.ms < 20.0);
    var ref = PlainCost{};
    var walked: f32 = PRIOR_PLAIN_MS;
    for ([_]f32{ 400, 13.0, 13.5, 14.0 }) |ms| {
        ref.sample(ms);
        walked += 0.1 * (std.math.clamp(ms, walked * 0.8, walked * 1.25) - walked);
        try approx(ref.ms, walked);
    }
}

test "PlainCost.seeded (DSpark): the first samples replace the prior, the fastest of them kept" {
    var c = PlainCost.seeded;
    c.sample(13.0);
    try approx(c.ms, 13.0);
    c.sample(19.0); // a slow tick among the seeds
    c.sample(12.2);
    try approx(c.ms, 12.2);
    c.sample(30.0); // seeded: back to the slow, bounded walk
    try testing.expect(c.ms > 12.2 and c.ms < 12.6);
    var stalled = PlainCost.seeded;
    for ([_]f32{ 400, 13.0, 13.5 }) |ms| stalled.sample(ms);
    try approx(stalled.ms, 13.0);
}

test "PlainCost.seeded (DSpark): a machine far slower than the prior is learned" {
    var c = PlainCost.seeded;
    for (0..80) |_| c.sample(120.0);
    try approx(c.ms, 120.0);
}

test "Drafting: a slow round moves the average wall by its full share" {
    var d = Drafting{};
    for (0..4) |_| d.record(30, 20);
    d.record(30, 2000);
    try approx(d.wall, 20 + 0.3 * 1980);
    try testing.expect(!d.shouldDraft());
}

test "CopyStats.rowsFor: the rows chooseRows picks for a flat survival, t1 alone when the match has failed" {
    const c = CopyStats{};
    const costs = priorCosts(17.7);
    const p: [8]f32 = @splat(c.probability(20));
    try testing.expectEqual(chooseRows(&p, &costs).rows, c.rowsFor(20, 8, 17.7));
    try testing.expect(c.rowsFor(20, 8, 17.7) > 1);
    try testing.expectEqual(@as(u32, 1), c.rowsFor(20, 0, 17.7));
    var failing = CopyStats{};
    for (0..40) |_| failing.observe(20, 8, 0);
    try testing.expectEqual(@as(u32, 1), failing.rowsFor(20, 8, 17.7));
    try testing.expectEqual(c.rowsFor(20, MAX_DRAFTS, 17.7), c.rowsFor(20, 100, 17.7));
}

test "CopyStats: a strong match starts trusted, and misses teach it otherwise" {
    var c = CopyStats{};
    try testing.expect(c.probability(20) > c.probability(8));
    for (0..12) |_| c.observe(20, 14, 0);
    try testing.expect(c.probability(20) < 0.3);
    try approx(c.probability(8), 0.7); // the other class is untouched
    var d = CopyStats{};
    for (0..12) |_| d.observe(20, 14, 14);
    try testing.expect(d.probability(20) > 0.9);
}

test "Drafting: a losing draft owes plain rounds, doubling to a cap; a winning probe resets" {
    var d = Drafting{};
    try testing.expect(d.shouldDraft());
    d.record(7, 10); // loses: one plain round owed
    try testing.expect(!d.shouldDraft());
    try testing.expect(d.shouldDraft());
    d.record(7, 10); // loses again: two owed
    try testing.expect(!d.shouldDraft());
    try testing.expect(!d.shouldDraft());
    try testing.expect(d.shouldDraft());
    var guard: u32 = 0;
    // However long it loses, the wait stops growing at 32 plain rounds.
    while (guard < 12) : (guard += 1) {
        d.record(5, 10);
        var owed: u32 = 0;
        while (!d.shouldDraft()) owed += 1;
        try testing.expect(owed <= 32);
    }
    d.record(5, 10);
    var owed: u32 = 0;
    while (!d.shouldDraft()) owed += 1;
    try testing.expectEqual(@as(u32, 32), owed);
    // A probe that pays clears the debt.
    d.record(60, 10);
    try testing.expect(d.shouldDraft());
    try testing.expect(d.probes > 0 and d.skipped > 0);
}

test "Drafting: a margin keeps drafting through rounds a little under break-even, not through junk" {
    var thin: Drafting = .{ .margin = 0.9 };
    for (0..8) |_| {
        try testing.expect(thin.shouldDraft());
        thin.record(21, 22.4); // 0.94 of its cost: under break-even, inside the margin
    }
    var strict: Drafting = .{};
    try testing.expect(strict.shouldDraft());
    strict.record(21, 22.4);
    try testing.expect(!strict.shouldDraft());
    var junk: Drafting = .{ .margin = 0.9 };
    try testing.expect(junk.shouldDraft());
    junk.record(17.6, 22.4); // nothing kept: the plain step's tokens for a draft's cost on top
    try testing.expect(!junk.shouldDraft());
}

test "Drafting: a thin win with scattered misses keeps drafting, because the rounds are weighed by their time" {
    var d = Drafting{};
    // 1.6x on average, a quarter of the rounds below break-even: the mean of the ratios would dip under 1.
    const rounds = [_][2]f32{ .{ 70, 40 }, .{ 18, 40 }, .{ 70, 40 }, .{ 70, 40 }, .{ 18, 40 }, .{ 120, 45 }, .{ 18, 40 }, .{ 120, 45 }, .{ 70, 40 }, .{ 18, 40 }, .{ 70, 40 }, .{ 120, 45 } };
    for (rounds) |r| {
        try testing.expect(d.shouldDraft());
        d.record(r[0], r[1]);
    }
}

test "Drafting: one failed round among winners does not stop drafting" {
    var d = Drafting{};
    for (0..6) |_| {
        try testing.expect(d.shouldDraft());
        d.record(25, 10);
    }
    try testing.expect(d.shouldDraft());
    d.record(6, 10);
    try testing.expect(d.shouldDraft());
}

test "PositionStats: conditional acceptance counts only the positions a round reached" {
    var st = PositionStats{};
    st.record(7, 7); // every position reached and kept
    st.record(7, 2); // 0 and 1 kept, 2 rejected, 3.. never reached
    st.record(0, 0); // a plain round
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("1.00/1.00/0.50/1.00/1.00/1.00/1.00", st.formatAcc(&buf));
    try testing.expectEqualStrings("1:1,8:2", st.formatRows(&buf));
}

test "PositionStats: the widest copied chain records without overflow" {
    var st = PositionStats{};
    st.record(15, 15);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("16:1", st.formatRows(&buf));
}

test "PositionStats: a position no round reached is not listed" {
    var st = PositionStats{};
    st.record(3, 0);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("0.00", st.formatAcc(&buf));
    st.record(5, 5);
    try testing.expectEqualStrings("0.50/1.00/1.00/1.00/1.00", st.formatAcc(&buf));
}

test "chooseGroupRows: rows go to the likeliest drafts across slots, none when the group's rows do not pay" {
    var cost: GroupCost = .{};
    const c2 = cost.costs(2);
    var drafts: [2]u32 = undefined;
    // A sure copy and an unsure stage draft: the sure slot takes the rows.
    const sure: [8]f32 = @splat(0.97);
    const weak = [_]f32{ 0.2, 0.2, 0.2, 0.2, 0.2 };
    _ = chooseGroupRows(&.{ &sure, &weak }, &c2, &drafts);
    try testing.expect(drafts[0] >= 6);
    try testing.expectEqual(@as(u32, 0), drafts[1]);
    // Neither pays: the group's plain step.
    const tokens = chooseGroupRows(&.{ &weak, &weak }, &c2, &drafts);
    try testing.expectEqual([2]u32{ 0, 0 }, drafts);
    try approx(tokens, 2);
    // Two sure slots share the row budget, the total never past it.
    const c4 = cost.costs(4);
    var d4: [4]u32 = undefined;
    _ = chooseGroupRows(&.{ &sure, &sure, &sure, &sure }, &c4, &d4);
    var total: u32 = 4;
    for (d4) |d| total += d;
    try testing.expect(total <= MAX_GROUP_ROWS);
    try testing.expect(@max(d4[0], d4[3]) - @min(d4[0], d4[3]) <= 1);
}

test "chooseGroupRows: a poisoned probability is no acceptance, and the cut stays a prefix of each slot's drafts" {
    var cost: GroupCost = .{};
    const c = cost.costs(1);
    var drafts: [1]u32 = undefined;
    _ = chooseGroupRows(&.{&[_]f32{ 0.99, std.math.nan(f32), 0.99, 0.99 }}, &c, &drafts);
    try testing.expect(drafts[0] <= 1);
}

test "Drafting: a stalled round is not what a round costs" {
    // A round that emits four tokens in the time of two plain steps.
    const gain: f32 = 4 * 17.7;
    var d: Drafting = .{};
    for (0..20) |_| {
        try testing.expect(d.shouldDraft());
        d.record(gain, 35);
    }
    // A kernel's first compile, a foreign GPU job: one round takes five seconds. Nothing is owed for it.
    d.record(gain, 5000);
    for (0..20) |_| {
        try testing.expect(d.shouldDraft());
        d.record(gain, 35);
    }
    // The first round of all stalls: a few plain rounds, then drafting again.
    var cold: Drafting = .{};
    cold.record(gain, 5000);
    var rounds: usize = 0;
    while (cold.skipped < 100 and rounds < 8) {
        if (cold.shouldDraft()) {
            cold.record(gain, 35);
            rounds += 1;
        }
    }
    try testing.expect(cold.skipped < 20);
    try testing.expect(cold.shouldDraft());
}

test "GroupCost: rows and states cost more, and a slower machine's rounds scale the prior within bounds" {
    var cost: GroupCost = .{};
    try testing.expect(cost.ms(2, 4) > cost.ms(2, 2) and cost.ms(3, 3) > cost.ms(2, 2));
    for (0..200) |_| cost.sample(2, 4, 2 * GroupCost.prior(2, 4));
    try testing.expectApproxEqAbs(@as(f32, 2), cost.scale, 0.05);
    cost.sample(2, 4, std.math.nan(f32));
    cost.sample(2, 4, 1e9);
    try testing.expect(cost.scale <= 2.5);
}

test "Calibration: a head prior reads the drafter's confidence as its acceptance until verifies say otherwise" {
    var head: Calibration = .{ .head = true };
    const fitted: Calibration = .{};
    try testing.expectApproxEqAbs(@as(f32, 0.93), head.probability(0.93), 1e-4);
    try testing.expect(fitted.probability(0.93) < 0.7);
    for (0..60) |_| head.observe(0.93, false);
    try testing.expect(head.probability(0.93) < 0.2);
}
