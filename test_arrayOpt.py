# -*- coding: utf-8 -*-
"""
Tests for arrayOpt.py

Verifies:
   1. Gradient computation via jax.grad
   2. Short Adam optimizer run
   3. Chained DE -> Adam optimization
   4. Chained Adam -> Adam warm-start continuation
   5. 3D position plot
   6. z-bound penalty keeps positions within volume_extent_z (loss without "unlimited")
   7. Mirror aggregation ("mean"/"max") matches the individual mirror residuals
   8. PSO optimizer respects its hard position bounds
   9. broadband residual_limits penalty and budget cost_limit behavior
  10. Geometric invariants of state_to_coordinates() across coordinate modes
  11. writeFile2() / ReadData2() round-trip
  12. stop_event stops Adam / DE / PSO early
  13. Mirror-reflection symmetry with a symmetric array geometry
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")          # headless — avoids display issues in CI/server
import matplotlib.pyplot as plt
import threading
import time
import tempfile
import os

import arrayOpt as AO

# ──────────────────────────────────────────────────────────────────────────────
# Shared test parameters
# ──────────────────────────────────────────────────────────────────────────────

N      = 2          # seismometers
freq   = 10.0       # Hz
SNR    = 15.0
p      = 0.3        # P-wave fraction
mirror = "mean"      # which mirror(s) the residual is reported for
loss   = "residual"  # loss_function type ("residual"/"broadband"/"budget")

np.random.seed(42)
state = np.random.uniform(-200, 200, N * 3)   # flat array, volume mode


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _make_jax():
    return AO.AnalyticResidual(default_mode="volume", default_mirror=mirror)


def _section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Gradient computation
# ──────────────────────────────────────────────────────────────────────────────

def test_gradient():
    _section("Test 1: Gradient via jax.grad")

    ar = _make_jax()

    def loss_fn(s):
        return ar.residual(s, N, freq, SNR, p, mirror=mirror)

    grad_fn = jax.grad(jax.jit(loss_fn))
    grad = grad_fn(jnp.array(state))

    print(f"  Gradient shape : {grad.shape}")
    print(f"  Gradient norm  : {float(jnp.linalg.norm(grad)):.6f}")
    print(f"  Gradient[:6]   : {np.array(grad[:6]).round(6)}")

    assert grad.shape == (N * 3,), f"Unexpected gradient shape: {grad.shape}"
    assert jnp.all(jnp.isfinite(grad)), "Gradient contains non-finite values"
    assert jnp.linalg.norm(grad) > 0, "Gradient is identically zero"
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Short Adam run (standalone)
# ──────────────────────────────────────────────────────────────────────────────

def test_adam_standalone():
    _section("Test 2: Adam optimizer (50 steps, N=2)")

    ar = _make_jax()
    best_res, best_pos = ar.optimize_Adam(
        N, freq, SNR, p,
        mirror=mirror, loss=loss,
        optimization_options={"max_steps": 50, "learning_rate": 1e-3, "tolerance_steps": 10},
        initial_state=state,
    )
    # best_res is the (boundary-penalized) loss_function value, not the raw
    # residual — only residual() itself is guaranteed to lie in [0, 1].
    actual_residual = float(ar.residual(best_pos, N, freq, SNR, p, mirror=mirror))

    print(f"  Best loss (penalized) : {best_res:.6f}")
    print(f"  Actual residual       : {actual_residual:.6f}")
    print(f"  Best position         : {np.array(best_pos).round(2)}")

    assert np.isfinite(best_res), "Adam returned non-finite loss"
    assert actual_residual <= 1.0 + 1e-6, f"Residual > 1 makes no physical sense: {actual_residual}"
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Chained DE -> Adam
# ──────────────────────────────────────────────────────────────────────────────

def test_chain_de_adam():
    _section("Test 3: Chained DE -> Adam")

    ar = _make_jax()

    chain = [
        ("DE",   {"niter": 5, "popsize": 5, "ftol": 1e-2}),
        ("Adam", {"max_steps": 30, "learning_rate": 1e-3, "tolerance_steps": 10}),
    ]

    final_res, final_pos = ar.optimize_chain(N, freq, SNR, p, chain, mirror=mirror, loss=loss)

    print(f"  Final residual : {final_res:.6f}")
    print(f"  Final position : {np.array(final_pos).round(2)}")

    assert np.isfinite(final_res), "Chain returned non-finite residual"
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Chained Adam -> Adam (warm-start continuation)
# ──────────────────────────────────────────────────────────────────────────────

def test_chain_adam_adam():
    _section("Test 4: Chained Adam -> Adam (warm-start)")

    ar = _make_jax()

    # First pass: coarse
    res1, pos1 = ar.optimize_Adam(
        N, freq, SNR, p, mirror=mirror, loss=loss,
        optimization_options={"max_steps": 20, "learning_rate": 1e-3, "tolerance_steps": 5},
        initial_state=state,
    )

    # Second pass: refinement from first result
    chain = [
        ("Adam", {"max_steps": 20, "learning_rate": 1e-4, "tolerance_steps": 5}),
    ]
    res2, pos2 = ar.optimize_chain(N, freq, SNR, p, chain, mirror=mirror, loss=loss)
    # (chain starts from None here; test that we can also pass initial state explicitly)
    res3, pos3 = ar.optimize_Adam(
        N, freq, SNR, p, mirror=mirror, loss=loss,
        optimization_options={"max_steps": 20, "learning_rate": 1e-4, "tolerance_steps": 5},
        initial_state=pos1,
    )

    print(f"  Residual after coarse Adam : {res1:.6f}")
    print(f"  Residual after fine   Adam : {res3:.6f}")

    assert np.isfinite(res3), "Warm-start Adam returned non-finite residual"
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: 3D plot
# ──────────────────────────────────────────────────────────────────────────────

def test_plot():
    _section("Test 5: 3D position plot")

    ar = _make_jax()
    fig, ax = ar.new_state_plot_3D()
    ar.plot_state_3D(ax, state, N)

    assert fig is not None
    plt.close(fig)
    print("  Plot created and closed without error")
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: z-bound penalty enforcement
# ──────────────────────────────────────────────────────────────────────────────

def test_zbound_enforced():
    _section("Test 6: z-bound penalty keeps positions within volume_extent_z")

    # Tight z-bounds and a start close to the upper one, so a few hundred Adam
    # steps are enough for the optimizer to actually press against the wall.
    zbound = 50.0
    ar = _make_jax()
    ar.defineGeometry(volume_extent_z=[-zbound, zbound])
    near_bound_state = np.array([0.0, 0.0, zbound - 10, 0.0, 0.0, -(zbound - 10)])

    opts = {"max_steps": 300, "learning_rate": 1.0, "tolerance_steps": 350}

    _, pos_constrained = ar.optimize_Adam(
        N, freq, SNR, p, mirror=mirror, loss=loss,  # loss="residual" — no "unlimited"
        optimization_options=opts, initial_state=near_bound_state,
    )
    z_constrained = np.array(pos_constrained).reshape(N, 3)[:, 2]

    _, pos_unlimited = ar.optimize_Adam(
        N, freq, SNR, p, mirror=mirror, loss=loss + " unlimited",
        optimization_options=opts, initial_state=near_bound_state,
    )
    z_unlimited = np.array(pos_unlimited).reshape(N, 3)[:, 2]

    print(f"  z (loss={loss!r})            : {z_constrained.round(3)}")
    print(f"  z (loss={loss + ' unlimited'!r}) : {z_unlimited.round(3)}")

    assert np.all(z_constrained >= -zbound - 1e-2) and np.all(z_constrained <= zbound + 1e-2), (
        f"z escaped volume_extent_z bounds [-{zbound}, {zbound}] despite penalized loss: {z_constrained}"
    )
    # Without the penalty, the same push is free to cross the same wall —
    # confirms the bound above is actually due to the penalty, not the physics.
    assert np.any(np.abs(z_unlimited) > zbound + 1e-2), (
        "expected 'unlimited' loss to allow crossing the z bound for comparison"
    )
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: mirror aggregation self-consistency
# ──────────────────────────────────────────────────────────────────────────────

def test_mirror_aggregation():
    _section("Test 7: mirror aggregation (mean/max) matches individual mirrors")

    ar = _make_jax()

    # combine_in=True: residual() internally averages/maxes over [in, end1, end2]
    r_in   = float(ar.residual(state, N, freq, SNR, p, mirror="in",   combine_in=True))
    r_end1 = float(ar.residual(state, N, freq, SNR, p, mirror="end1", combine_in=True))
    r_end2 = float(ar.residual(state, N, freq, SNR, p, mirror="end2", combine_in=True))
    r_mean = float(ar.residual(state, N, freq, SNR, p, mirror="mean", combine_in=True))
    r_max  = float(ar.residual(state, N, freq, SNR, p, mirror="max",  combine_in=True))
    r_all  = float(ar.residual(state, N, freq, SNR, p, mirror="all",  combine_in=True))

    # residual() returns sqrt(Res_v_entry); the "mean"/"max" modes aggregate the
    # *pre-sqrt* Res_v values and take sqrt once at the end. Since sqrt is
    # monotonic, max(sqrt(a,b,c)) == sqrt(max(a,b,c)) — but mean does NOT commute
    # with sqrt, so the individual-mirror ground truth must be un-sqrt'd first.
    expected_mean = float(np.sqrt(np.mean(np.square([r_in, r_end1, r_end2]))))
    expected_max  = float(np.max([r_in, r_end1, r_end2]))

    print(f"  combine_in=True : in={r_in:.6f} end1={r_end1:.6f} end2={r_end2:.6f}")
    print(f"  mean: got={r_mean:.6f} expected(sqrt of mean of squares of in/end1/end2)={expected_mean:.6f}")
    print(f"  max : got max={r_max:.6f} all={r_all:.6f} expected(max of in/end1/end2)={expected_max:.6f}")

    assert abs(r_mean - expected_mean) < 1e-9, (
        f"mirror='mean' (combine_in=True) returned {r_mean}, but sqrt(mean(in^2, end1^2, end2^2)) = "
        f"{expected_mean} — residual()'s aggregation over Res_v looks wrong for combine_in=True."
    )
    assert abs(r_max - expected_max) < 1e-9, (
        f"mirror='max' (combine_in=True) returned {r_max}, but max(in, end1, end2) = {expected_max}."
    )
    assert abs(r_all - expected_max) < 1e-9, (
        f"mirror='all' (combine_in=True) returned {r_all}, expected it to equal mirror='max' = {expected_max}."
    )

    # combine_in=False: residual() internally averages/maxes over [in1, end1, end2, in2]
    r_in1   = float(ar.residual(state, N, freq, SNR, p, mirror="in1",  combine_in=False))
    r_in2   = float(ar.residual(state, N, freq, SNR, p, mirror="in2",  combine_in=False))
    r_end1f = float(ar.residual(state, N, freq, SNR, p, mirror="end1", combine_in=False))
    r_end2f = float(ar.residual(state, N, freq, SNR, p, mirror="end2", combine_in=False))
    r_meanf = float(ar.residual(state, N, freq, SNR, p, mirror="mean", combine_in=False))
    r_maxf  = float(ar.residual(state, N, freq, SNR, p, mirror="max",  combine_in=False))

    expected_meanf = float(np.sqrt(np.mean(np.square([r_in1, r_end1f, r_end2f, r_in2]))))
    expected_maxf  = float(np.max([r_in1, r_end1f, r_end2f, r_in2]))

    print(f"  combine_in=False: in1={r_in1:.6f} end1={r_end1f:.6f} end2={r_end2f:.6f} in2={r_in2:.6f}")
    print(f"  mean: got={r_meanf:.6f} expected(sqrt of mean of squares)={expected_meanf:.6f}")
    print(f"  max : got={r_maxf:.6f} expected={expected_maxf:.6f}")

    assert abs(r_meanf - expected_meanf) < 1e-9, (
        f"mirror='mean' (combine_in=False) returned {r_meanf}, but "
        f"sqrt(mean(in1^2, end1^2, end2^2, in2^2)) = {expected_meanf}."
    )
    assert abs(r_maxf - expected_maxf) < 1e-9, (
        f"mirror='max' (combine_in=False) returned {r_maxf}, but "
        f"max(in1, end1, end2, in2) = {expected_maxf}."
    )

    # combine_in is an approximation, so it need not change the result — just
    # confirm both settings run cleanly and give a finite, physically valid residual.
    for label, val in [("combine_in=True mean", r_mean), ("combine_in=False mean", r_meanf)]:
        assert np.isfinite(val) and 0.0 <= val <= 1.0 + 1e-6, (
            f"{label} = {val} is not a finite residual in [0, 1]."
        )
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: PSO respects its hard bounds
# ──────────────────────────────────────────────────────────────────────────────

def test_pso_standalone():
    _section("Test 8: PSO optimizer respects hard bounds")

    ar = _make_jax()
    ar.defineGeometry(volume_extent_z=[-50, 50], volume_extent_xy=[-100, 100])

    best_res, best_pos = ar.optimize_PSO(
        N, freq, SNR, p, mirror=mirror, loss=loss,
        optimization_options={"swarm_size": 20, "niter": 10, "k": 5},
    )
    pos = np.array(best_pos).reshape(N, 3)
    lb, ub = np.array(ar.lower_bound), np.array(ar.upper_bound)

    print(f"  best_res : {best_res:.6f}")
    print(f"  position : {pos.round(2).tolist()}")
    print(f"  bounds   : lower={lb.tolist()} upper={ub.tolist()}")

    assert np.isfinite(best_res), f"PSO returned a non-finite loss: {best_res}"
    assert np.all(pos >= lb - 1e-6) and np.all(pos <= ub + 1e-6), (
        f"PSO position escaped its hard bounds: pos={pos.tolist()}, "
        f"bounds=(lower={lb.tolist()}, upper={ub.tolist()})."
    )
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: broadband residual_limits penalty and budget cost_limit
# ──────────────────────────────────────────────────────────────────────────────

def test_broadband_and_budget_loss():
    _section("Test 9: broadband residual_limits penalty and budget cost_limit")

    ar = _make_jax()
    freqs = np.array([5.0, 10.0])
    test_state = np.array([0., 0., 0., 50., 50., 50.])

    # broadband: exceeding residual_limits should sharply inflate the loss
    limits_easy = np.ones(2)         # never violated: residual() is always <= 1
    limits_hard = np.full(2, 1e-6)   # always violated

    loss_easy   = float(ar.loss_function(test_state, N, freqs, SNR, p, loss="broadband unlimited", residual_limits=limits_easy))
    loss_hard   = float(ar.loss_function(test_state, N, freqs, SNR, p, loss="broadband unlimited", residual_limits=limits_hard))
    loss_direct = float(ar.broadband_loss(test_state, N, freqs, SNR, p, residual_limits=limits_easy))

    print(f"  broadband loss, easy limits (never violated) : {loss_easy:.4f}")
    print(f"  broadband loss, hard limits (always violated): {loss_hard:.4f}  (expected >> easy)")
    print(f"  loss_function('broadband unlimited') vs broadband_loss() direct : {loss_easy:.6f} vs {loss_direct:.6f}")

    assert np.isclose(loss_easy, loss_direct), (
        f"loss_function(loss='broadband unlimited') = {loss_easy} should exactly match "
        f"broadband_loss() called directly = {loss_direct} (no other penalty is active)."
    )
    assert loss_hard > 10 * loss_easy, (
        f"exceeding residual_limits should sharply inflate the broadband loss, but "
        f"hard-limits loss={loss_hard:.4f} is not >> easy-limits loss={loss_easy:.4f}."
    )

    # budget: cost_limit must be a no-op unless "budget" is also present in loss
    # (a fresh instance is used here since `ar` above is already JIT-locked, and
    # defineGeometry() refuses to mutate a locked instance — see AnalyticResidual's
    # _jit_cache_key()/_check_not_jit_locked() docstrings)
    ar_budget = _make_jax()
    ar_budget.defineGeometry(volume_extent_z=[-50, 50])
    deep_state = np.array([0., 0., -50., 0., 0., -50.])  # both seismometers at max depth

    loss_no_budget    = float(ar_budget.loss_function(deep_state, N, freq, SNR, p, loss="residual unlimited", cost_limit=0.5))
    loss_budget_tight = float(ar_budget.loss_function(deep_state, N, freq, SNR, p, loss="residual unlimited budget", cost_limit=0.5))
    loss_budget_loose = float(ar_budget.loss_function(deep_state, N, freq, SNR, p, loss="residual unlimited budget", cost_limit=5.0))

    print(f"  cost_limit=0.5, 'budget' absent  : {loss_no_budget:.4f}  (cost_limit should be ignored)")
    print(f"  cost_limit=0.5, 'budget' present : {loss_budget_tight:.4f}  (over budget -> expected >> above)")
    print(f"  cost_limit=5.0, 'budget' present : {loss_budget_loose:.4f}  (under budget -> expected ~= no-budget case)")

    assert np.isclose(loss_no_budget, loss_budget_loose, rtol=1e-3), (
        f"a generous cost_limit ({loss_budget_loose:.4f}) should behave like no budget constraint at all "
        f"({loss_no_budget:.4f}) since the depth budget isn't exceeded — got a mismatch."
    )
    assert loss_budget_tight > 100 * loss_no_budget, (
        f"cost_limit=0.5 with 'budget' in loss should sharply penalize this over-budget state, but "
        f"tight-budget loss={loss_budget_tight:.4f} is not >> unconstrained loss={loss_no_budget:.4f}."
    )
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 10: geometric invariants across coordinate modes
# ──────────────────────────────────────────────────────────────────────────────

def test_coordinate_mode_invariants():
    _section("Test 10: geometric invariants of state_to_coordinates() across modes")

    np.random.seed(1)

    # sphere: every point should sit at radius == tunnel_radius from the origin
    ar_sphere = AO.AnalyticResidual(default_mode="sphere")
    s = np.random.uniform(0, 200, N * ar_sphere.dim)
    x, y, z, Nloc = ar_sphere.state_to_coordinates(s, N)
    r = np.sqrt(np.array(x) ** 2 + np.array(y) ** 2 + np.array(z) ** 2)
    print(f"  sphere radius   : {r.round(6)} (expected all == {ar_sphere.tunnel_radius})")
    assert np.allclose(r, ar_sphere.tunnel_radius, atol=1e-8), (
        f"'sphere' mode: points should all be at radius {ar_sphere.tunnel_radius}, got radii {r}."
    )

    # cylinder: every point should sit at radial distance tunnel_radius from the x-axis
    ar_cyl = AO.AnalyticResidual(default_mode="cylinder")
    s = np.random.uniform(0, 300, N * ar_cyl.dim)
    x, y, z, Nloc = ar_cyl.state_to_coordinates(s, N)
    r = np.sqrt(np.array(y) ** 2 + np.array(z) ** 2)
    print(f"  cylinder radius : {r.round(6)} (expected all == {ar_cyl.tunnel_radius})")
    assert np.allclose(r, ar_cyl.tunnel_radius, atol=1e-8), (
        f"'cylinder' mode: points should all be at radius {ar_cyl.tunnel_radius} from the x-axis, got {r}."
    )

    # 2cylinder: perpendicular distance from the phi-selected arm should be tunnel_radius
    ar_2cyl = AO.AnalyticResidual(default_mode="2cylinder")
    phi = np.array([30., -45., 90., -120.])[:N]
    along = np.random.uniform(0, 300, N)
    s2 = np.stack([phi, along], axis=1).ravel()
    x, y, z, Nloc = ar_2cyl.state_to_coordinates(s2, N)
    pos = np.stack([np.array(x), np.array(y), np.array(z)], axis=1)
    perp_norms = np.zeros(N)
    for i in range(N):
        arm = np.array(ar_2cyl.e1 if phi[i] > 0 else ar_2cyl.e2)
        perp_norms[i] = np.linalg.norm(pos[i] - np.dot(pos[i], arm) * arm)
    print(f"  2cylinder perp. distance from arm: {perp_norms.round(6)} (expected all == {ar_2cyl.tunnel_radius})")
    assert np.allclose(perp_norms, ar_2cyl.tunnel_radius, atol=1e-6), (
        f"'2cylinder' mode: perpendicular distance from the phi-selected arm should be "
        f"{ar_2cyl.tunnel_radius}, got {perp_norms}."
    )

    # volume forcesym: Nloc doubles, y mirrors, z stays the same
    ar_fs = AO.AnalyticResidual(default_mode="volume forcesym")
    s = np.random.uniform(-100, 100, N * ar_fs.dim)
    x, y, z, Nloc = ar_fs.state_to_coordinates(s, N)
    x, y, z = np.array(x), np.array(y), np.array(z)
    print(f"  forcesym Nloc: {Nloc} (expected {2 * N})")
    assert Nloc == 2 * N, f"'volume forcesym' should double Nloc to {2 * N}, got {Nloc}."
    assert np.allclose(y[:N], -y[N:]), (
        f"'volume forcesym': the mirrored half's y should be the negative of the first half's y, "
        f"got y[:N]={y[:N]} vs -y[N:]={-y[N:]}."
    )
    assert np.allclose(z[:N], z[N:]), (
        f"'volume forcesym' (no 'mirror'): z should be identical between the two halves, "
        f"got z[:N]={z[:N]} vs z[N:]={z[N:]}."
    )

    # volume forcesym mirror: z also flips between the two halves
    ar_fsm = AO.AnalyticResidual(default_mode="volume forcesym mirror")
    x, y, z, Nloc = ar_fsm.state_to_coordinates(s, N)
    z = np.array(z)
    assert np.allclose(z[:N], -z[N:]), (
        f"'volume forcesym mirror': z should be negated between the two halves, "
        f"got z[:N]={z[:N]} vs -z[N:]={-z[N:]}."
    )

    # volume multipleX: Nloc == N * X, and Nmult reflects X
    ar_mult = AO.AnalyticResidual(default_mode="volume multiple3")
    s = np.random.uniform(-100, 100, N * ar_mult.dim)
    x, y, z, Nloc = ar_mult.state_to_coordinates(s, N)
    print(f"  multiple3 Nloc: {Nloc} (expected {N * 3}), dim={ar_mult.dim}, Nmult={ar_mult.Nmult}")
    assert Nloc == N * 3, f"'volume multiple3' should give Nloc={N * 3}, got {Nloc}."
    assert ar_mult.Nmult == 3, f"'volume multiple3' should set Nmult=3, got {ar_mult.Nmult}."

    # Smoke test: residual() stays finite in every supported mode. This is a plain
    # forward-only call (a few seconds to compile per mode) — cheap enough to run
    # for all modes. A full jax.grad(jax.jit(...)) differentiability check is much
    # more expensive to compile (tens of seconds to low minutes *per mode*, since
    # each mode is a distinct closure with nothing to share a JIT cache with), so
    # that's only worth doing for one mode here — "volume" is already covered by
    # Test 1, so we single out "2cylinder" (the trickiest control flow: it
    # branches per-seismometer on sign(phi) via boolean masking) as the one most
    # likely to reveal a differentiability issue if one existed.
    modes = ["volume", "sphere", "cylinder", "2cylinder", "2cylindervolume",
             "volume forcesym", "volume forcesym mirror", "volume multiple3"]
    grad_checked_mode = "2cylinder"
    for m in modes:
        ar_m = AO.AnalyticResidual(default_mode=m)
        s = np.random.uniform(1, 50, N * ar_m.dim)  # avoid 0 to dodge angle/radius edge cases

        val = float(ar_m.residual(jnp.array(s), N, freq, SNR, p, mirror="mean"))
        ok = np.isfinite(val)

        grad_note = ""
        if m == grad_checked_mode:
            def mode_loss(st, ar_m=ar_m):
                return ar_m.residual(st, N, freq, SNR, p, mirror="mean")

            grad = jax.grad(jax.jit(mode_loss))(jnp.array(s))
            ok = ok and bool(jnp.all(jnp.isfinite(grad)))
            grad_note = f" finite_grad={ok}"

        print(f"  mode={m!r:28s} residual={val:.6f}{grad_note}")
        assert ok, f"mode={m!r}: residual/gradient contains non-finite values (residual={val})."
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 11: writeFile2 / ReadData2 round-trip
# ──────────────────────────────────────────────────────────────────────────────

def test_writefile2_roundtrip():
    _section("Test 11: writeFile2 / ReadData2 round-trip")

    ar = _make_jax()
    test_state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    test_residual = 0.123456

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "result")
        ar.writeFile2(test_residual, test_state, N, freq, SNR, p, path, starttime=0)
        rd = AO.ReadData2(path + ".txt")

        checks = [
            ("mode", rd.mode, ar.mode),
            ("N",    rd.N,    N),
            ("f",    rd.f,    freq),
            ("SNR",  rd.SNR,  SNR),
            ("p",    rd.p,    p),
            ("loss", rd.loss, ar.loss),
        ]
        for label, got, expected in checks:
            print(f"  {label:6s}: got={got!r} expected={expected!r}")
            assert got == expected, f"ReadData2.{label} = {got!r}, expected {expected!r} (round-trip mismatch)."

        print(f"  state   : got={rd.state.tolist()} expected={test_state.tolist()}")
        assert np.allclose(rd.state, test_state), (
            f"ReadData2.state = {rd.state.tolist()} does not match the written state {test_state.tolist()}."
        )

        print(f"  residual: got={rd.residual} expected={test_residual}")
        assert np.isclose(rd.residual, test_residual), (
            f"ReadData2.residual = {rd.residual}, expected {test_residual}."
        )
        assert np.isclose(rd.energy, test_residual), (
            f"ReadData2.energy (alias for residual) = {rd.energy}, expected {test_residual}."
        )

        count = AO.ReadData2.count(path + ".txt")
        print(f"  record count: {count} (expected 1)")
        assert count == 1, f"expected exactly 1 record in a freshly written file, got {count}."
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 12: stop_event stops Adam / DE / PSO early
# ──────────────────────────────────────────────────────────────────────────────

def test_stop_event():
    _section("Test 12: stop_event stops Adam / DE / PSO early")

    ar = _make_jax()
    already_stopped = threading.Event()
    already_stopped.set()

    # Adam: with stop_event already set, the gradient-step loop body never runs,
    # so the returned position must be exactly the initial_state.
    _, adam_pos = ar.optimize_Adam(
        N, freq, SNR, p, mirror=mirror, loss=loss,
        optimization_options={"max_steps": 1000, "learning_rate": 1.0, "tolerance_steps": 999999},
        initial_state=state, stop_event=already_stopped,
    )
    unchanged = np.allclose(adam_pos, state)
    print(f"  Adam: returned position unchanged from initial_state: {unchanged}")
    assert unchanged, (
        f"Adam should not take any step with a pre-set stop_event, but returned "
        f"position {np.array(adam_pos)} != initial state {state}."
    )

    # DE: with stop_event already set, the scipy callback aborts after the very
    # first generation, so this must return quickly despite a huge niter.
    t0 = time.time()
    de_res, _ = ar.optimize_DE(
        N, freq, SNR, p, mirror=mirror, loss=loss,
        optimization_options={"niter": 100000, "popsize": 10}, stop_event=already_stopped,
    )
    elapsed = time.time() - t0
    print(f"  DE: stopped after {elapsed:.1f}s despite niter=100000 (best_res={de_res:.6f})")
    assert elapsed < 30, (
        f"DE with a pre-set stop_event took {elapsed:.1f}s — it should abort after its first "
        f"generation instead of running toward niter=100000."
    )

    # PSO: with stop_event already set, _PSOStopped fires before any particle is
    # evaluated, so this returns the documented sentinel (cost=inf, pos=zeros).
    pso_res, pso_pos = ar.optimize_PSO(
        N, freq, SNR, p, mirror=mirror, loss=loss,
        optimization_options={"niter": 100000, "swarm_size": 20, "k": 5}, stop_event=already_stopped,
    )
    print(f"  PSO: best_res={pso_res} (expected inf), best_pos={np.array(pso_pos)} (expected zeros)")
    assert pso_res == float("inf"), (
        f"PSO with a pre-set stop_event should report cost=inf (no particle was ever evaluated), "
        f"got {pso_res}."
    )
    assert np.allclose(pso_pos, 0.0), (
        f"PSO with a pre-set stop_event should return the zero-vector sentinel position, "
        f"got {np.array(pso_pos)}."
    )
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Test 13: mirror-reflection symmetry
# ──────────────────────────────────────────────────────────────────────────────

def test_reflection_symmetry():
    _section("Test 13: mirror-reflection symmetry with a symmetric array geometry")

    # e1_sym/e2_sym are mirror images of each other across the x-axis (y -> -y),
    # so reflecting the seismometer array the same way should leave mirror="mean"/
    # "max" unchanged, while individual end1/end2 residuals should swap.
    ar = AO.AnalyticResidual(default_mode="volume", default_mirror="mean", e1=AO.e1_sym, e2=AO.e2_sym)

    np.random.seed(7)
    s = np.random.uniform(-200, 200, N * 3)
    reflected = s.reshape(N, 3).copy()
    reflected[:, 1] *= -1
    reflected = reflected.ravel()

    for m in ("mean", "max"):
        r_orig = float(ar.residual(s, N, freq, SNR, p, mirror=m))
        r_refl = float(ar.residual(reflected, N, freq, SNR, p, mirror=m))
        print(f"  mirror={m!r}: orig={r_orig:.8f} reflected={r_refl:.8f} diff={abs(r_orig - r_refl):.2e}")
        assert abs(r_orig - r_refl) < 1e-8, (
            f"with symmetric geometry (e1_sym/e2_sym), mirror={m!r} should be invariant under "
            f"a y -> -y reflection of the whole seismometer array, but orig={r_orig} != reflected={r_refl}."
        )

    # end1/end2 individually should *swap* under the reflection (not match themselves)
    r_end1_orig = float(ar.residual(s, N, freq, SNR, p, mirror="end1"))
    r_end2_refl = float(ar.residual(reflected, N, freq, SNR, p, mirror="end2"))
    print(f"  end1(orig)={r_end1_orig:.8f} vs end2(reflected)={r_end2_refl:.8f}")
    assert abs(r_end1_orig - r_end2_refl) < 1e-8, (
        f"end1 on the original array should equal end2 on the y-reflected array (the reflection "
        f"swaps which physical mirror each seismometer array is closer to), but "
        f"{r_end1_orig} != {r_end2_refl}."
    )
    print("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_gradient,
        test_adam_standalone,
        test_chain_de_adam,
        test_chain_adam_adam,
        test_plot,
        test_zbound_enforced,
        test_mirror_aggregation,
        test_pso_standalone,
        test_broadband_and_budget_loss,
        test_coordinate_mode_invariants,
        test_writefile2_roundtrip,
        test_stop_event,
        test_reflection_symmetry,
    ]

    passed = []
    failed = []

    for test_fn in tests:
        try:
            test_fn()
            passed.append(test_fn.__name__)
        except Exception as e:
            failed.append((test_fn.__name__, e))
            print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"  Results: {len(passed)}/{len(tests)} passed")
    if failed:
        print("  Failed tests:")
        for name, err in failed:
            print(f"    {name}: {err}")
    else:
        print("  All tests passed.")
    print("=" * 60)
