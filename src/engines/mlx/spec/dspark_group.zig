//! DeepSeek-V4.1 DSpark rounds for several slots of one model at once: each slot drafts (its context's continuation,
//! or the stages, every slot's block in one pass), the group's verify rows are priced together
//! (`dflash_policy.chooseGroupRows`), one batched verify runs them, and each slot accepts under its own sampling and
//! keeps its accepted prefix. A slot alone ticks its solo round (`nextDspark`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const env = @import("../../../env.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const dsv41_mod = @import("../../../plugins/deepseek_v41/deepseek_v41.zig");
const dflash_policy = @import("dflash/policy.zig");
const intree = @import("../plugin/intree.zig");
const modules = @import("../modules/call.zig");
const dspark = @import("dspark.zig");
const Generator = @import("../generate/generator.zig").Generator;
const DrafterStepResult = @import("drafter.zig").DrafterStepResult;
const Transformer = @import("../transformer.zig").Transformer;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Scheduler = @import("../scheduler.zig").Scheduler;
const Slot = @import("../slot.zig").Slot;
const finishSlot = @import("../finish.zig").finishSlot;
const runSingleDecodeTick = @import("../decode.zig").runSingleDecodeTick;
const loopGuardTick = @import("../loop_stop.zig").loopGuardTick;
const publishSpeculativeBlock = @import("wiring.zig").publishSpeculativeBlock;
const slotSpecMode = @import("wiring.zig").slotSpecMode;
const capAcceptedForTokenBudget = @import("verify.zig").capAcceptedForTokenBudget;
const dsparkCopyLen = @import("verify.zig").dsparkCopyLen;
const specDecodeUnsupported = @import("verify.zig").specDecodeUnsupported;
const SamplingParams = @import("../sampling/params.zig").SamplingParams;
const sampleRowsBegin = @import("../sampling/lazy.zig").sampleRowsBegin;
const sampleRowsRead = @import("../sampling/lazy.zig").sampleRowsRead;

/// Slots one group round carries.
pub const MAX_SLOTS: usize = dflash_policy.MAX_GROUP_ROWS;

var stage_slots_max: ?usize = null;

/// The most slots a group round still drafts with the stages for (`MLX_SERVE_DSV41_GROUP_DRAFT_MAX`, 3). A plain batched
/// step whose first layers run behind the step before it (`dsv41_mod.leadBatched`) costs 2-3% less than one the host
/// builds after its read, and a row bought beside three other requests costs 5 ms against 3.9 alone: on an M5 Ultra,
/// greedy, 4 different requests decode 120 tok/s on plain steps against 113-117 drafting (long prose; 117 against
/// 112-115 on short mixed tasks), 3 requests 105 against 102-104 on prose and 104 against 105-107 on the tasks, 2
/// requests 89 against 92 and 87 against 93. From 5 the plain step won before the lead (8 requests: 135 against 129).
/// A copy from the context costs no draft pass and is drafted at any count.
fn stageSlotsMax() usize {
    if (stage_slots_max == null) stage_slots_max = if (env.get(.MLX_SERVE_DSV41_GROUP_DRAFT_MAX)) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 3 else 3;
    return stage_slots_max.?;
}

/// Whether `xfm` runs DSpark group rounds: a DeepSeek-V4.1 module with its stages that batches its states (resident,
/// unpaired). Its DSpark slots keep drafting beside company instead of yielding to the plain batch.
pub fn groupable(xfm: *const Transformer) bool {
    const m = intree.dsv41(xfm) orelse return false;
    return m.dspark != null and m.pair == null and m.streamed == null;
}

/// Whether `slot` ticks DSpark rounds on a groupable model.
pub fn slotGroupable(slot: *const Slot) bool {
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (gen.spec_disabled_runtime or slotSpecMode(slot, gen) != .dspark) return false;
    return groupable(slot.model.transformer orelse return false);
}

/// The decode tick's DSpark slots on groupable models: one group round per model, `MAX_SLOTS` slots at a time; a slot
/// alone ticks solo.
pub fn runDsparkGroups(sch: *Scheduler, slots: []*Slot) !void {
    std.sort.pdq(*Slot, slots, {}, struct {
        fn lt(_: void, a: *Slot, b: *Slot) bool {
            return @intFromPtr(a.model) < @intFromPtr(b.model);
        }
    }.lt);
    var start: usize = 0;
    while (start < slots.len) {
        var end = start + 1;
        while (end < slots.len and slots[end].model == slots[start].model) end += 1;
        var g0 = start;
        while (g0 < end) {
            const sub = slots[g0..@min(end, g0 + MAX_SLOTS)];
            if (sub.len == 1) try runSingleDecodeTick(sch, sub[0]) else try runGroupTick(sch, sub);
            g0 += sub.len;
        }
        start = end;
    }
}

fn runGroupTick(sch: *Scheduler, group: []*Slot) !void {
    var inner_err: ?anyerror = null;
    runGroupTickInner(sch, group) catch |e| {
        inner_err = e;
    };
    mlx.checkErrorDecode() catch |mlx_err| {
        log.err("[scheduler] DSpark group round aborted: MLX failure mid-generation ({s}) — failing these requests, the server keeps serving\n", .{@errorName(mlx_err)});
        for (group) |slot| slot.markError(@errorName(mlx_err));
        return;
    };
    if (inner_err) |e| return e;
}

fn runGroupTickInner(sch: *Scheduler, group: []*Slot) !void {
    var live: [MAX_SLOTS]*Slot = undefined;
    var gens: [MAX_SLOTS]*Generator = undefined;
    var allocs: [MAX_SLOTS]std.mem.Allocator = undefined;
    var n: usize = 0;
    for (group) |slot| {
        const gen = &slot.legacy_gen.?;
        if (try loopGuardTick(sch, slot, gen)) continue;
        const entry = enter(gen, slot.allocator) catch |e| {
            slot.markError(@errorName(e));
            continue;
        };
        switch (entry) {
            .ready => {
                live[n] = slot;
                gens[n] = gen;
                allocs[n] = slot.allocator;
                n += 1;
            },
            .emitted => |tok| publishSpeculativeBlock(sch, slot, gen, &.{tok}),
            .done => finishSlot(sch, slot, gen.finish_reason),
        }
    }
    if (n == 0) return;
    if (n == 1) return runSingleDecodeTick(sch, live[0]);
    var results: [MAX_SLOTS]DrafterStepResult = undefined;
    try modules.call("dspark_group_round", .{ gens[0..n], allocs[0..n], results[0..n] });
    for (live[0..n], gens[0..n], results[0..n]) |slot, gen, r| {
        defer slot.allocator.free(r.tokens);
        publishSpeculativeBlock(sch, slot, gen, r.tokens);
    }
}

const Entry = union(enum) { ready, emitted: u32, done };

/// `nextDspark`'s way in: a pipelined token still in flight is published first (the slot joins the next round), a
/// stopped request ends; else the slot stands at the spec invariant.
fn enter(gen: *Generator, allocator: std.mem.Allocator) !Entry {
    if (gen.done) return .done;
    if (specDecodeUnsupported(gen.sampling, gen.logprobs_n)) return error.SpecDecodeUnsupported;
    if (gen.has_pending_logits or gen.has_pending_token) switch (try gen.drainPipelineForSpec(allocator)) {
        .drained => |tok| return .{ .emitted = tok },
        .stopped => return .done,
        .already_clean, .stay_disabled => return error.PipelineHalfState,
    };
    if (try gen.checkStop()) return .done;
    return .ready;
}

const Kind = enum { plain, copy, stage };

/// One DSpark round for each of `gens` (slots of one DeepSeek-V4.1 model, each at the spec invariant): its drafts, the
/// verify rows the group's pricing buys, one batched verify, its accept under its own sampling, its kept prefix.
/// `out[k]` is `gens[k]`'s committed block, its tokens owned through `allocs[k]`.
pub fn moduleGroupRound(gens: []const *Generator, allocs: []const std.mem.Allocator, out: []DrafterStepResult) !void {
    const n = gens.len;
    const xfm = gens[0].xfm;
    const mdl = intree.dsv41(xfm).?;
    const gpa = allocs[0];
    const led0 = mdl.ds_group_led;
    const lost0 = mdl.ds_group_lost;
    var ctxs: [MAX_SLOTS]*ForwardCtx = undefined;
    for (gens, ctxs[0..n]) |g, *c| c.* = &g.ctx;
    var states: [MAX_SLOTS]*dsv41_mod.State = undefined;
    var ids: [MAX_SLOTS]u64 = undefined;
    if (!intree.dsv41States(xfm, ctxs[0..n], states[0..n], ids[0..n])) return error.Dsv41BatchUnsupported;

    // Drafts: the context's continuation where it holds one, else the stages for every slot that drafts, in one pass.
    var kinds: [MAX_SLOTS]Kind = undefined;
    var drafts: [MAX_SLOTS]dsv41_mod.DsparkDraft = undefined;
    var caps: [MAX_SLOTS]u32 = undefined;
    var suffix: [MAX_SLOTS]u32 = @splat(0);
    var made: usize = 0;
    defer for (drafts[0..made]) |*d| d.deinit(gpa);
    var staged: [MAX_SLOTS]usize = undefined;
    var n_stage: usize = 0;
    for (gens, 0..) |gen, k| {
        const t1 = gen.next_token_id;
        caps[k] = capAcceptedForTokenBudget(std.math.maxInt(u32), gen.completion_tokens, gen.max_tokens);
        const idx = try gen.mtpLookupIndex(allocs[k]);
        const found = idx.matchBest(t1, @intCast(dsv41_mod.maxVerifyDrafts(mdl)));
        const copy = dsparkCopyLen(found, gen.max_tokens -| gen.completion_tokens -| 1, mdl.ds_block);
        // A group with no row to spare verifies no stage draft: it drafts none.
        const stage = copy == 0 and n <= stageSlotsMax() and n < dflash_policy.MAX_GROUP_ROWS and if (gen.dspark_pol) |*pol| pol.drafting.shouldDraft() else true;
        kinds[k] = if (copy > 0) .copy else if (stage) .stage else .plain;
        if (stage) {
            staged[n_stage] = k;
            n_stage += 1;
        }
        const ids_k = try gpa.alloc(u32, copy + 1);
        ids_k[0] = t1;
        if (copy > 0) {
            @memcpy(ids_k[1..], found.?.draft[0..copy]);
            suffix[k] = found.?.suffix;
        }
        drafts[k] = .{ .ids = ids_k, .len = copy, .logits = &.{}, .confidence = &.{} };
        made += 1;
    }
    var draft_ms: f32 = 0;
    if (n_stage > 0) {
        // The module times the draft: one it launched behind the last step has run since that step ended.
        defer draft_ms = @as(f32, @floatFromInt(mdl.ds_draft_ns)) / std.time.ns_per_ms;
        var sts: [MAX_SLOTS]*dsv41_mod.State = undefined;
        var t1s: [MAX_SLOTS]u32 = undefined;
        for (staged[0..n_stage], 0..) |k, i| {
            sts[i] = states[k];
            t1s[i] = gens[k].next_token_id;
        }
        var blocks: [MAX_SLOTS]dsv41_mod.DsparkDraft = undefined;
        // Every slot's chunk starts with its committed token, whatever the group then buys: the verify's first layers
        // on those rows go to the GPU behind the draft.
        var firsts: [MAX_SLOTS]u32 = undefined;
        for (gens, firsts[0..n]) |g, *t| t.* = g.next_token_id;
        try dsv41_mod.dsparkDraftBatched(mdl, gpa, sts[0..n_stage], t1s[0..n_stage], false, blocks[0..n_stage], .{ .states = states[0..n], .gens = ids[0..n], .t1s = firsts[0..n] });
        for (staged[0..n_stage], blocks[0..n_stage]) |k, b| {
            drafts[k].deinit(gpa);
            drafts[k] = b;
        }
    }

    // The rows the group buys: each slot's drafts priced by its acceptance (the confidence head's through the slot's
    // calibration, a copy's survival), never past its token budget.
    var probs: [MAX_SLOTS][dflash_policy.MAX_DRAFTS]f32 = undefined;
    var p: [MAX_SLOTS][]const f32 = undefined;
    for (gens, drafts[0..n], kinds[0..n], 0..) |gen, d, kind, k| {
        const avail = @min(d.len, caps[k], dflash_policy.MAX_DRAFTS);
        for (probs[k][0..avail], 0..) |*o, j| o.* = switch (kind) {
            .stage => if (gen.dspark_pol) |*pol| pol.calib.probability(dspark.sigmoid(d.confidence[j])) else 1,
            .copy => if (gen.dspark_pol) |*pol| pol.copy.probability(suffix[k]) else 1,
            .plain => unreachable,
        };
        p[k] = probs[k][0..avail];
    }
    const cost = &xfm.dspark_group_cost;
    const costs = cost.costs(n);
    var rows_k: [MAX_SLOTS]u32 = undefined;
    _ = dflash_policy.chooseGroupRows(p[0..n], &costs, rows_k[0..n]);
    var chunks: [MAX_SLOTS][]const u32 = undefined;
    var rows: usize = 0;
    for (drafts[0..n], rows_k[0..n], chunks[0..n]) |*d, b, *ch| {
        d.len = b;
        ch.* = d.ids[0 .. b + 1];
        rows += b + 1;
    }

    // One verify: a round that verifies no draft is the group's plain step, with nothing to roll back.
    const verify_watch = io_util.Stopwatch.init(gens[0].timer.io);
    const drafted = rows > n;
    const logits = try dsv41_mod.extendBatched(mdl, gpa, states[0..n], ids[0..n], chunks[0..n], drafted);
    defer _ = mlx.mlx_array_free(logits);
    for (gens) |g| {
        g.dspark_led += mdl.ds_group_led - led0;
        g.dspark_lost += mdl.ds_group_lost - lost0;
    }
    // Each slot's accept on its own rows. A slot's `t1` alone draws its next token with the others' in one batch, as the
    // plain batched tick does.
    var accepted: [MAX_SLOTS]u32 = @splat(0);
    var next: [MAX_SLOTS]u32 = undefined;
    var offs: [MAX_SLOTS]usize = undefined;
    var draw_rows: [MAX_SLOTS]mlx.mlx_array = undefined;
    var draw_params: [MAX_SLOTS]SamplingParams = undefined;
    var draw_slot: [MAX_SLOTS]usize = undefined;
    var draw_of: [MAX_SLOTS]usize = undefined;
    var n_draw: usize = 0;
    defer for (draw_rows[0..n_draw]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    var greedy = false;
    var off: usize = 0;
    for (gens, chunks[0..n], 0..) |gen, ch, k| {
        offs[k] = off;
        off += ch.len;
        if (ch.len > 1) {
            greedy = greedy or !gen.dspark_stochastic;
            continue;
        }
        const row = try rowsOf(logits, offs[k], 1, mdl.vocab, mdl.s);
        defer _ = mlx.mlx_array_free(row);
        var r = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_reshape(&r, row, &[_]c_int{ 1, 1, @intCast(mdl.vocab) }, 3, mdl.s));
        draw_rows[n_draw] = r;
        draw_params[n_draw] = gen.sampling;
        draw_slot[n_draw] = k;
        draw_of[k] = n_draw;
        n_draw += 1;
    }
    // What the GPU makes of the rows before the host reads anything: the lone tokens' draws, and every row's own token
    // for the greedy slots.
    const draws: mlx.mlx_array = if (n_draw > 0) try sampleRowsBegin(draw_rows[0..n_draw], draw_params[0..n_draw], mdl.s) else .{ .ctx = null };
    defer if (draws.ctx != null) {
        _ = mlx.mlx_array_free(draws);
    };
    const am: mlx.mlx_array = if (greedy) try argmaxLazy(logits, mdl.s) else .{ .ctx = null };
    defer if (am.ctx != null) {
        _ = mlx.mlx_array_free(am);
    };
    // Both are one batch of their own: a wait on either ends with the verify, not with what is queued behind it.
    {
        var both: [2]mlx.mlx_array = undefined;
        var n_both: usize = 0;
        for ([_]mlx.mlx_array{ draws, am }) |a| if (a.ctx != null) {
            both[n_both] = a;
            n_both += 1;
        };
        if (n_both > 0) {
            const vec = mlx.mlx_vector_array_new_data(&both, n_both);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_async_eval(vec));
        }
    }
    // A sampled slot's verdict on its drafts goes to the GPU too, read further down: an evaluation begun after
    // something is queued behind the verify would wait for that to run.
    var sampled: [MAX_SLOTS]?dspark.Sampled = @splat(null);
    defer for (&sampled) |*sv| if (sv.*) |*x| x.deinit();
    for (gens, chunks[0..n], 0..) |gen, ch, k| if (ch.len > 1 and gen.dspark_stochastic) {
        const rows_l = try rowsOf(logits, offs[k], ch.len, mdl.vocab, mdl.s);
        defer _ = mlx.mlx_array_free(rows_l);
        sampled[k] = try dspark.stochasticBegin(gen, rows_l, ch, mdl.vocab, caps[k], true);
    };
    // The round after this one, as far as it can be told now, goes to the GPU behind this one's verdict: a plain step's
    // first layers (`dsv41_mod.leadBatched`), or the next draft (`dsparkDraftAheadBatched`, `dsparkDraftAheadVerify`).
    const draft_share = if (n_stage > 0) draft_ms / @as(f32, @floatFromInt(n_stage)) else 0;
    const stages = n <= stageSlotsMax() and n < dflash_policy.MAX_GROUP_ROWS;
    if (!drafted and plainNext(gens, caps[0..n], kinds[0..n], draft_share)) {
        try dsv41_mod.leadBatched(mdl, gpa, states[0..n], ids[0..n], draws);
    } else if (stages) ahead: {
        // The slots due a stage draft next round. A slot whose drafts are in this verify is weighed by what its rows
        // are expected to keep; the round's end records what they kept.
        const token_ms = cost.ms(n, n) / @as(f32, @floatFromInt(n));
        var due: [MAX_SLOTS]usize = undefined;
        var n_due: usize = 0;
        for (gens, 0..) |gen, k| {
            var expect: f32 = 0;
            var alive: f32 = 1;
            for (p[k][0..rows_k[k]]) |pj| {
                alive *= pj;
                expect += alive;
            }
            const spent = draft_share + @as(f32, @floatFromInt(rows_k[k])) * cost.rowMs();
            if (caps[k] == 0 or !stagesNext(gen, kinds[k], expect * token_ms, spent)) continue;
            if (drafted and !dsv41_mod.aheadFitsVerify(mdl, states[k])) break :ahead;
            // A rule whose verdict the GPU does not pick (TokenV3, a keyed request's serial one) is the host's alone.
            if (sampled[k]) |sv| if (sv.lazy == null) break :ahead;
            due[n_due] = k;
            n_due += 1;
        }
        if (n_due == 0) break :ahead;
        var due_states: [MAX_SLOTS]*dsv41_mod.State = undefined;
        var due_ids: [MAX_SLOTS]u64 = undefined;
        for (due[0..n_due], due_states[0..n_due], due_ids[0..n_due]) |k, *st, *id| {
            st.* = states[k];
            id.* = ids[k];
        }
        // Each due slot's verdict, still the GPU's: its draw, or the drafts its rows' own tokens agree with.
        var verdicts: [MAX_SLOTS]dsv41_mod.LazyVerdict = undefined;
        var picked: usize = 0;
        defer for (verdicts[0..picked]) |v| {
            _ = mlx.mlx_array_free(v.kept);
            _ = mlx.mlx_array_free(v.next);
        };
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        if (draws.ctx != null) {
            var shaped = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(shaped);
            try mlx.check(mlx.mlx_reshape(&shaped, draws, &[_]c_int{@intCast(n_draw)}, 1, mdl.s));
            try mlx.check(mlx.mlx_astype(&flat, shaped, .int32, mdl.s));
        }
        for (due[0..n_due]) |k| {
            if (sampled[k]) |sv| {
                var kept = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(kept);
                try mlx.check(mlx.mlx_array_set(&kept, sv.lazy.?[0]));
                var after = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(after);
                try mlx.check(mlx.mlx_array_set(&after, sv.lazy.?[1]));
                verdicts[picked] = .{ .kept = kept, .next = after };
            } else if (chunks[k].len > 1) {
                verdicts[picked] = try dsv41_mod.greedyVerdictRows(mdl, gpa, am, offs[k], chunks[k], caps[k]);
            } else {
                var one = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(one);
                const at: c_int = @intCast(draw_of[k]);
                try mlx.check(mlx.mlx_slice(&one, flat, &[_]c_int{at}, 1, &[_]c_int{at + 1}, 1, &[_]c_int{1}, 1, mdl.s));
                verdicts[picked] = .{ .kept = mlx.mlx_array_new_int(0), .next = one };
            }
            picked += 1;
        }
        if (drafted) {
            try dsv41_mod.dsparkDraftAheadVerify(mdl, due_states[0..n_due], due_ids[0..n_due], verdicts[0..n_due]);
        } else {
            var nexts: [MAX_SLOTS]mlx.mlx_array = undefined;
            for (verdicts[0..n_due], nexts[0..n_due]) |v, *t| t.* = v.next;
            const vec = mlx.mlx_vector_array_new_data(&nexts, n_due);
            defer _ = mlx.mlx_vector_array_free(vec);
            var mine = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(mine);
            try mlx.check(mlx.mlx_concatenate_axis(&mine, vec, 0, mdl.s));
            try dsv41_mod.dsparkDraftAheadBatched(mdl, due_states[0..n_due], due_ids[0..n_due], mine);
        }
    }
    // The rows' own tokens and the draws, each a batch of its own where nothing above made one: a wait on either ends
    // with the verify, not with what was queued behind it.
    const best: []const i32 = if (greedy) try hostInts(gpa, am) else &.{};
    defer gpa.free(best);
    // The first read that waited for the step marks when it ended (`dsv41_mod.verdictRead`).
    var unmarked = !greedy;
    if (greedy) dsv41_mod.verdictRead(mdl);
    for (gens, chunks[0..n], 0..) |gen, ch, k| {
        if (ch.len == 1) continue;
        if (sampled[k]) |*sv| {
            const v = try dspark.stochasticFinish(gen, sv);
            if (unmarked) dsv41_mod.verdictRead(mdl);
            unmarked = false;
            accepted[k] = v.accepted;
            next[k] = v.next_token;
        } else {
            var a: u32 = 0;
            while (a + 1 < ch.len and @as(u32, @intCast(best[offs[k] + a])) == ch[a + 1]) a += 1;
            accepted[k] = @min(a, caps[k]);
            next[k] = @intCast(best[offs[k] + accepted[k]]);
        }
    }
    if (n_draw > 0) {
        var drawn: [MAX_SLOTS]i32 = undefined;
        try sampleRowsRead(drawn[0..n_draw], draws);
        if (unmarked) dsv41_mod.verdictRead(mdl);
        // The step begun behind this one goes on through the layers that need its tokens, before this round's
        // bookkeeping: its first layers run for less time than that takes.
        if (n_draw == n and dsv41_mod.groupLeadStands(mdl)) {
            var toks: [MAX_SLOTS]u32 = undefined;
            for (toks[0..n], drawn[0..n]) |*t, d| t.* = @intCast(d);
            try dsv41_mod.leadBatchedOn(mdl, gpa, states[0..n], ids[0..n], toks[0..n]);
        }
        for (draw_slot[0..n_draw], drawn[0..n_draw], draw_params[0..n_draw]) |k, t, sp| {
            next[k] = @intCast(t);
            gens[k].sampling.draw = sp.draw;
        }
    }
    cost.sample(n, rows, msOf(verify_watch.read()));

    // Each slot keeps its prefix, and teaches its drafting backoff what drafting cost the round (its share of the draft,
    // its rows) against what its accepted drafts saved (the group's plain step a token).
    var done: usize = 0;
    errdefer for (out[0..done], allocs[0..done]) |r, a| a.free(r.tokens);
    for (gens, states[0..n], drafts[0..n], kinds[0..n], 0..) |gen, st, d, kind, k| {
        if (drafted) try dsv41_mod.finishVerify(mdl, gpa, st, accepted[k] + 1);
        const tokens = try allocs[k].dupe(u32, d.ids[0 .. accepted[k] + 1]);
        out[k] = (try gen.commitDsparkRound(allocs[k], .{ .tokens = tokens, .accepted = accepted[k], .next_token = next[k] }, st.n)).?;
        done += 1;
        switch (kind) {
            .stage => {
                if (gen.dspark_pol) |*pol| for (d.confidence[0..d.len], 0..) |c, j| {
                    if (j > accepted[k]) break;
                    pol.calib.observe(dspark.sigmoid(c), j < accepted[k]);
                };
                gen.dspark_stats.record(@intCast(d.len), accepted[k]);
            },
            .copy => {
                gen.dspark_copies += 1;
                gen.dspark_copy_drafted += d.len;
                gen.dspark_copy_accepted += accepted[k];
                if (gen.dspark_pol) |*pol| pol.copy.observe(suffix[k], @intCast(d.len), accepted[k]);
            },
            .plain => gen.dspark_plain += 1,
        }
        gen.dspark_group_rounds += 1;
    }
    const token_ms = cost.ms(n, n) / @as(f32, @floatFromInt(n));
    for (gens, kinds[0..n], accepted[0..n], rows_k[0..n]) |gen, kind, a, b| if (kind == .stage) if (gen.dspark_pol) |*pol| {
        const spent = draft_ms / @as(f32, @floatFromInt(n_stage)) + @as(f32, @floatFromInt(b)) * cost.rowMs();
        pol.drafting.record(@as(f32, @floatFromInt(a)) * token_ms, spent);
    };
}

/// Whether a slot drafts from the stages in the round after this one, as this round can tell before its verdict is
/// read. A slot that did not stage this round stages once its backoff owes no more plain rounds; one that staged
/// records, at the round's end, what its drafts kept against what drafting cost the round (`spent`): `gain` stands
/// for that (nothing where no draft of its was bought, else what its rows are expected to keep).
fn stagesNext(gen: *const Generator, kind: Kind, gain: f32, spent: f32) bool {
    const pol = if (gen.dspark_pol) |*pp| pp else return true;
    if (kind != .stage) return pol.drafting.owed == 0;
    var after = pol.drafting;
    after.record(gain, spent);
    return after.owed == 0;
}

/// Whether the round after a round that bought no draft is expected to be a plain step for every one of `gens`
/// (`caps[k]`: the tokens slot k may still take after this round's): no slot is at its budget's end, and none is due a
/// stage draft (a group past `stageSlotsMax` drafts none; `spent`: a staged slot's share of this round's draft). A
/// copy from the context or a stop is not foreseen: the step begun for that round then serves the copy's first rows,
/// or is dropped (`dsv41_mod.leadBatched`).
fn plainNext(gens: []const *Generator, caps: []const u32, kinds: []const Kind, spent: f32) bool {
    const n = gens.len;
    const stages = n <= stageSlotsMax() and n < dflash_policy.MAX_GROUP_ROWS;
    for (gens, caps, kinds) |gen, cap, kind| {
        if (cap == 0) return false;
        if (stages and stagesNext(gen, kind, 0, spent)) return false;
    }
    return true;
}

/// Rows [lo, lo + n) of `logits` `[R, V]` (caller frees).
fn rowsOf(logits: mlx.mlx_array, lo: usize, n: usize, vocab: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_slice(&o, logits, &[_]c_int{ @intCast(lo), 0 }, 2, &[_]c_int{ @intCast(lo + n), @intCast(vocab) }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return o;
}

/// Every row's own token of `logits` `[R, V]`: `[R]` int32, not evaluated (caller frees).
fn argmaxLazy(logits: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, logits, -1, false, s));
    var am32 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(am32);
    try mlx.check(mlx.mlx_astype(&am32, am, .int32, s));
    return am32;
}

/// An int32 array's values on the host (caller frees).
fn hostInts(gpa: std.mem.Allocator, a: mlx.mlx_array) ![]i32 {
    try mlx.check(mlx.mlx_array_eval(a));
    const data = mlx.mlx_array_data_int32(a) orelse return error.NoData;
    return gpa.dupe(i32, data[0..mlx.mlx_array_size(a)]);
}

fn msOf(ns: u64) f32 {
    return @as(f32, @floatFromInt(ns)) / std.time.ns_per_ms;
}
