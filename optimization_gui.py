# -*- coding: utf-8 -*-
"""
Tkinter GUI for arrayOpt.AnalyticResidual optimization.

Features
--------
- Left panel: all optimization parameters (N, freq, SNR, p, mirror, mode,
  combine_in, optimizer + per-optimizer sub-options, n parallel runs).
- Right panel top: live 3-D scatter — mirrors as blue circles, seismometers
  as diamonds (black for a single run; tab10-coloured per-run for N>1).
  All runs shown simultaneously.
- Right panel mid: residual history — one coloured line per run.
- Right panel bottom: status label + Run / Stop buttons.

The optimization runs in background threads (one per run); a queue drains via
root.after(100, ...) to update the plots without blocking Tk's event loop.
Plot redraws are throttled to ~6 fps to keep the GUI responsive.
"""

import queue
import threading
import time

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import os

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

import arrayOpt as AO

_COLORS = plt.cm.tab10.colors  # 10 distinct colours; cycled for >10 runs

_MIRROR_COMBINED   = ["mean", "max", "end1", "end2", "in"]
_MIRROR_UNCOMBINED = ["mean", "max", "end1", "end2", "in1", "in2"]

# alpha per mirror (in1, in2, end1, end2) for each `mirror` selection
_DIM = 0.15
_MIRROR_OPACITY = {
    "mean": [1.0,  1.0,  1.0,  1.0 ],
    "max":  [1.0,  1.0,  1.0,  1.0 ],
    "in":   [1.0,  1.0,  _DIM, _DIM],
    "in1":  [1.0,  _DIM, _DIM, _DIM],
    "in2":  [_DIM, 1.0,  _DIM, _DIM],
    "end1": [_DIM, _DIM, 1.0,  _DIM],
    "end2": [_DIM, _DIM, _DIM, 1.0 ],
}

_HELP_PARAMETERS = """\
N              Number of seismometers (or boreholes for "volume multipleX" mode).

freq           Frequency in Hz at which seismic noise cancellation is evaluated.

SNR            Signal-to-noise ratio of each seismometer.
               Higher values mean less electronic noise relative to the seismic signal.

p              P-wave fraction [0 - 1].
               p = 1 -> pure P-waves;  p = 0 -> pure S-waves.

mirror         Mirror(s) to optimize for:
                 mean         average residual over all mirrors
                 max          worst-case (maximum) over all mirrors
                 end1 / end2  a specific end mirror
                 in           combined input mirror (available when combine_in is checked)
                 in1 / in2    individual input mirrors (available when combine_in is unchecked)

mode           Coordinate system for seismometer positions:
                 volume                  free 3-D Cartesian placement
                 cylinder                one tunnel arm at fixed radius
                 2cylinder               both tunnel arms at fixed radius
                 2cylindervolume         both arms with variable radius
                 sphere                  spherical shell at fixed radius
                 volume forcesym         volume + mirrored copy at bisecting plane
                 volume forcesym mirror  volume + mirrored copy at bisecting line
                 volume multipleX        X seismometers per borehole at different depths

X              Seismometers stacked per borehole (only for "volume multipleX").

combine_in     If checked, the two input mirrors are treated as one correlated
               measurement.  If unchecked, they contribute independently."""

_HELP_OPTIMIZER = """\
method         Optimization algorithm:
                 Adam      gradient descent - fast but local (best as final step)
                 DE        Differential Evolution - slow global search
                 PSO       Particle Swarm Optimization - slow global search
                 DE->Adam  global DE search, then Adam local refinement
                 PSO->Adam global PSO search, then Adam local refinement

Adam options
  learning rate  Step size per gradient update.
                 Smaller = more careful but slower convergence.
  max steps      Maximum gradient steps before the run stops.
  tol steps      Stop early if the residual does not improve
                 over this many consecutive steps.

DE options
  niter          Number of DE generations.
  popsize        Candidate solutions per generation.
                 Larger = more thorough search but slower.

PSO options
  niter          Number of PSO iterations.
  swarm size     Number of particles in the swarm.
                 Larger = more thorough search but slower."""

_HELP_PARALLEL = """\
n runs     Number of independent optimizations launched simultaneously.
           Each run starts from a different random initial state.
           All runs are shown live in the plots; the best result is highlighted.
           Increasing n runs raises the chance of finding the global optimum."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _float(widget, default=0.0):
    try:
        return float(widget.get())
    except ValueError:
        return default


def _int(widget, default=1):
    try:
        return int(widget.get())
    except ValueError:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationGUI:

    def __init__(self, root):
        self.root = root
        root.title("Seismometer Array Optimizer")
        root.resizable(True, True)

        self._result_queue = queue.Queue()
        self._stop_event   = threading.Event()

        # Per-run state (populated in _on_run)
        self._run_histories    = {}   # {run_id: {"steps": [], "residuals": []}}
        self._run_best         = {}   # {run_id: (residual, state)}
        self._run_step_offsets = {}   # {run_id: int} cumulative step offset across stages
        self._run_stage_lines  = {}   # {run_id: [x_pos, ...]} x positions for dashed dividers
        self._n_active      = 0
        self._n_runs_total  = 0
        self._shared_ar     = None
        self._shared_N      = 0

        self._last_plot_time = 0.0

        self._build_layout()
        self._build_left_panel()
        self._build_right_panel()
        self._update_optimizer_subframes()

        self.root.after(100, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._stop_event.set()
        plt.close("all")
        self.root.destroy()

    # ── Layout skeleton ──────────────────────────────────────────────────────

    def _build_layout(self):
        self._left = ttk.Frame(self.root, padding=8)
        self._left.grid(row=0, column=0, sticky="nsew")

        self._right = ttk.Frame(self.root, padding=8)
        self._right.grid(row=0, column=1, sticky="nsew")

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

    # ── Left panel ───────────────────────────────────────────────────────────

    def _row(self, parent, label, widget_factory, row, **kw):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        w = widget_factory(parent, **kw)
        w.grid(row=row, column=1, sticky="ew", padx=(4, 0), pady=2)
        return w

    def _entry(self, parent, default=""):
        e = ttk.Entry(parent, width=10)
        e.insert(0, str(default))
        return e

    def _combo(self, parent, values, default=None):
        c = ttk.Combobox(parent, values=values, state="readonly", width=12)
        c.set(default if default is not None else values[0])
        return c

    def _show_help(self, title, text):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.resizable(False, False)
        ttk.Label(win, text=text, justify="left", font=("Courier", 9),
                  padding=12).pack()
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 10))
        win.grab_set()

    def _build_left_panel(self):
        p = self._left
        p.columnconfigure(1, weight=1)
        row = 0

        # ── Physics parameters ────────────────────────────────────────────
        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        row += 1
        ttk.Label(p, text="PARAMETERS", font=("", 9, "bold")).grid(
            row=row, column=0, sticky="w")
        ttk.Button(p, text="?", width=2,
                   command=lambda: self._show_help("Parameters", _HELP_PARAMETERS)
                   ).grid(row=row, column=1, sticky="e")
        row += 1

        self._N_entry    = self._row(p, "N",          self._entry, row, default=3);    row += 1
        self._freq_entry = self._row(p, "freq (Hz)",  self._entry, row, default=10.0); row += 1
        self._SNR_entry  = self._row(p, "SNR",        self._entry, row, default=15.0); row += 1
        self._p_entry    = self._row(p, "p",          self._entry, row, default=0.2);  row += 1

        self._mirror_combo = self._row(
            p, "mirror", self._combo, row,
            values=_MIRROR_COMBINED, default="mean")
        self._mirror_combo.bind("<<ComboboxSelected>>",
                              lambda _e: self._update_3d_plot())
        row += 1

        self._mode_combo = self._row(
            p, "mode", self._combo, row,
            values=["volume", "cylinder", "2cylinder", "2cylindervolume",
                    "sphere", "volume forcesym", "volume forcesym mirror",
                    "volume multipleX"],
            default="volume")
        self._mode_combo.bind("<<ComboboxSelected>>",
                              lambda _e: self._update_mode_subframes())
        row += 1

        # multipleX sub-options (hidden until "volume multipleX" is selected)
        self._multiple_frame = ttk.LabelFrame(p, text="multiple options", padding=4)
        self._multiple_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self._multiple_frame.columnconfigure(1, weight=1)
        self._multiple_x = self._row(self._multiple_frame, "X (depth levels)", self._entry, 0, default=2)
        self._multiple_frame.grid_remove()
        row += 1

        ttk.Label(p, text="combine_in").grid(row=row, column=0, sticky="w", pady=2)
        self._combine_in_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(p, variable=self._combine_in_var,
                        command=self._on_combine_in_toggle).grid(
            row=row, column=1, sticky="w", padx=(4, 0))
        row += 1

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1

        # ── Optimizer selection ───────────────────────────────────────────
        ttk.Label(p, text="OPTIMIZER", font=("", 9, "bold")).grid(
            row=row, column=0, sticky="w")
        ttk.Button(p, text="?", width=2,
                   command=lambda: self._show_help("Optimizer", _HELP_OPTIMIZER)
                   ).grid(row=row, column=1, sticky="e")
        row += 1

        self._opt_combo = self._row(
            p, "method", self._combo, row,
            values=["Adam", "DE", "PSO", "DE->Adam", "PSO->Adam"],
            default="Adam")
        self._opt_combo.bind("<<ComboboxSelected>>",
                             lambda _e: self._update_optimizer_subframes())
        row += 1

        # Adam sub-options
        self._adam_frame = ttk.LabelFrame(p, text="Adam options", padding=4)
        self._adam_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self._adam_frame.columnconfigure(1, weight=1)
        af = self._adam_frame
        self._lr_entry       = self._row(af, "learning rate", self._entry, 0, default=1)
        self._adam_steps     = self._row(af, "max steps",     self._entry, 1, default=500)
        self._adam_tol_steps = self._row(af, "tol steps",     self._entry, 2, default=50)
        row += 1

        # DE sub-options
        self._de_frame = ttk.LabelFrame(p, text="DE options", padding=4)
        self._de_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self._de_frame.columnconfigure(1, weight=1)
        df = self._de_frame
        self._de_niter   = self._row(df, "niter",   self._entry, 0, default=200)
        self._de_popsize = self._row(df, "popsize", self._entry, 1, default=15)
        row += 1

        # PSO sub-options
        self._pso_frame = ttk.LabelFrame(p, text="PSO options", padding=4)
        self._pso_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self._pso_frame.columnconfigure(1, weight=1)
        pf = self._pso_frame
        self._pso_niter      = self._row(pf, "niter",      self._entry, 0, default=100)
        self._pso_swarm_size = self._row(pf, "swarm size", self._entry, 1, default=400)
        row += 1

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1

        # ── Parallel runs ─────────────────────────────────────────────────
        ttk.Label(p, text="PARALLEL RUNS", font=("", 9, "bold")).grid(
            row=row, column=0, sticky="w")
        ttk.Button(p, text="?", width=2,
                   command=lambda: self._show_help("Parallel runs", _HELP_PARALLEL)
                   ).grid(row=row, column=1, sticky="e")
        row += 1

        ttk.Label(p, text="n runs").grid(row=row, column=0, sticky="w", pady=2)
        self._n_runs_spin = ttk.Spinbox(p, from_=1, to=32, width=6)
        self._n_runs_spin.set(1)
        self._n_runs_spin.grid(row=row, column=1, sticky="w", padx=(4, 0), pady=2)
        row += 1

    def _update_optimizer_subframes(self):
        method = self._opt_combo.get()
        adam_visible = method in ("Adam", "DE->Adam", "PSO->Adam")
        de_visible   = method in ("DE",   "DE->Adam")
        pso_visible  = method in ("PSO",  "PSO->Adam")

        if adam_visible: self._adam_frame.grid()
        else:            self._adam_frame.grid_remove()

        if de_visible:   self._de_frame.grid()
        else:            self._de_frame.grid_remove()

        if pso_visible:  self._pso_frame.grid()
        else:            self._pso_frame.grid_remove()

    def _update_mode_subframes(self):
        if self._mode_combo.get() == "volume multipleX":
            self._multiple_frame.grid()
        else:
            self._multiple_frame.grid_remove()

    def _on_combine_in_toggle(self):
        if self._combine_in_var.get():
            new_values = _MIRROR_COMBINED
            if self._mirror_combo.get() in ("in1", "in2"):
                self._mirror_combo.set("in")
        else:
            new_values = _MIRROR_UNCOMBINED
            if self._mirror_combo.get() == "in":
                self._mirror_combo.set("in1")
        self._mirror_combo.config(values=new_values)
        self._update_3d_plot()

    # ── Right panel ──────────────────────────────────────────────────────────

    def _build_right_panel(self):
        p = self._right
        p.columnconfigure(0, weight=1)
        p.rowconfigure(0, weight=3)
        p.rowconfigure(1, weight=2)
        p.rowconfigure(2, weight=0)

        self._fig3d = plt.figure(figsize=(5, 4))
        self._ax3d  = self._fig3d.add_subplot(111, projection="3d")
        self._ax3d.set_title("Seismometer positions")
        self._canvas3d = FigureCanvasTkAgg(self._fig3d, master=p)
        self._canvas3d.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._fig_res, self._ax_res = plt.subplots(figsize=(5, 2.5))
        self._ax_res.set_xlabel("step")
        self._ax_res.set_ylabel("residual")
        self._ax_res.set_title("Residual history")
        self._canvas_res = FigureCanvasTkAgg(self._fig_res, master=p)
        self._canvas_res.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        bottom = ttk.Frame(p, padding=4)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(bottom, textvariable=self._status_var, anchor="w").grid(
            row=0, column=0, sticky="ew")

        btn_frame = ttk.Frame(bottom)
        btn_frame.grid(row=1, column=0, sticky="e", pady=(4, 0))

        self._run_btn  = ttk.Button(btn_frame, text="Run",  command=self._on_run)
        self._run_btn.grid(row=0, column=0, padx=(0, 4))

        self._stop_btn = ttk.Button(btn_frame, text="Stop",
                                    command=self._on_stop, state="disabled")
        self._stop_btn.grid(row=0, column=1, padx=(0, 4))

        self._save_btn = ttk.Button(btn_frame, text="Save results",
                                    command=self._on_save, state="disabled")
        self._save_btn.grid(row=0, column=2)

    # ── Plot helpers ─────────────────────────────────────────────────────────

    def _update_3d_plot(self):
        if not self._run_best or self._shared_ar is None:
            return

        ar  = self._shared_ar
        N   = self._shared_N
        ax  = self._ax3d
        ax.cla()
        ax.set_title("Seismometer positions")

        best_id = min(self._run_best, key=lambda k: self._run_best[k][0])
        single  = (self._n_runs_total == 1)

        for run_id, (_, state) in self._run_best.items():
            try:
                x, y, z, _ = ar.state_to_coordinates(state, N)
                x = np.array(x); y = np.array(y); z = np.array(z)
            except Exception:
                continue

            color  = 'k' if single else _COLORS[run_id % len(_COLORS)]
            size   = 80 if (run_id == best_id) else 35
            label  = f"Run {run_id + 1}" if not single else "seismometers"
            ax.scatter(x, y, z, marker='D', color=color, s=size,
                       zorder=3, label=label)

        # Static mirror positions — blue circles, dimmed when not the active mirror
        try:
            mirror = self._mirror_combo.get()
            mirror_alphas = _MIRROR_OPACITY.get(mirror, [1.0, 1.0, 1.0, 1.0])
            for vec, alpha in zip(
                    (ar.d_in1 * ar.e1, ar.d_in2 * ar.e2,
                     ar.d_end1 * ar.e1, ar.d_end2 * ar.e2),
                    mirror_alphas):
                v = np.array(vec)
                if v.shape == (3,) and np.linalg.norm(v) > 0:
                    ax.scatter([v[0]], [v[1]], [v[2]],
                               marker='o', color='blue', s=100, zorder=2, alpha=alpha)
        except Exception:
            pass

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=7)
        self._canvas3d.draw_idle()

    def _update_residual_plot(self):
        ax = self._ax_res
        ax.cla()
        ax.set_xlabel("step")
        ax.set_ylabel("residual")
        ax.set_title("Residual history")
        single = (self._n_runs_total == 1)
        for run_id, hist in self._run_histories.items():
            if not hist["steps"]:
                continue
            color = 'k' if single else _COLORS[run_id % len(_COLORS)]
            ax.plot(hist["steps"], hist["residuals"],
                    color=color, linewidth=1,
                    label=f"Run {run_id + 1}" if not single else None)
            for x_div in self._run_stage_lines.get(run_id, []):
                ax.axvline(x=x_div, color=color, linestyle="--",
                           linewidth=0.8, alpha=0.5)
        if not single and self._run_histories:
            ax.legend(fontsize=7, loc="upper right")
        self._fig_res.tight_layout()
        self._canvas_res.draw_idle()

    def _maybe_redraw(self):
        now = time.monotonic()
        if now - self._last_plot_time >= 0.15:
            self._update_3d_plot()
            self._update_residual_plot()
            self._last_plot_time = now

    # ── Thread communication ─────────────────────────────────────────────────

    def _make_callback(self, run_id):
        def callback(step, residual, state):
            self._result_queue.put(("step", run_id, step, residual, state))
        return callback

    def _make_stage_callback(self, run_id):
        def stage_callback(text):
            self._result_queue.put(("stage", run_id, text))
        return stage_callback

    def _poll_queue(self):
        try:
            while True:
                msg = self._result_queue.get_nowait()

                if msg[0] == "step":
                    _, run_id, step, residual, state = msg
                    offset = self._run_step_offsets.get(run_id, 0)
                    abs_step = step + offset
                    self._run_histories[run_id]["steps"].append(abs_step)
                    self._run_histories[run_id]["residuals"].append(residual)
                    self._run_best[run_id] = (residual, state)
                    best_res = min(v[0] for v in self._run_best.values())
                    self._status_var.set(
                        f"Running ({self._n_active} active) — "
                        f"step {abs_step}  best residual={best_res:.5f}")
                    self._maybe_redraw()

                elif msg[0] == "stage":
                    _, run_id, text = msg
                    hist = self._run_histories.get(run_id, {"steps": [], "residuals": []})
                    if hist["steps"]:
                        # Record the boundary between this stage and the previous one.
                        last = hist["steps"][-1]
                        self._run_stage_lines[run_id].append(last + 0.5)
                        # Offset so the next stage's step 1 lands just after last.
                        self._run_step_offsets[run_id] = last
                    best_str = ""
                    if self._run_best:
                        best_res = min(v[0] for v in self._run_best.values())
                        best_str = f"  best so far={best_res:.5f}"
                    label = f"Run {run_id + 1}: {text}{best_str}" \
                            if self._n_runs_total > 1 else f"{text}{best_str}"
                    self._status_var.set(label)

                elif msg[0] == "done":
                    _, run_id, best_res, best_pos = msg
                    self._run_best[run_id] = (best_res, best_pos)
                    self._n_active -= 1
                    self._maybe_redraw()
                    if self._n_active == 0:
                        self._update_3d_plot()
                        self._update_residual_plot()
                        overall = min(v[0] for v in self._run_best.values())
                        self._run_btn.config(state="normal")
                        self._stop_btn.config(state="disabled")
                        self._save_btn.config(state="normal")
                        self._status_var.set(
                            f"Done — best residual = {overall:.6f} "
                            f"across {self._n_runs_total} run(s)")

                elif msg[0] == "error":
                    _, run_id, exc = msg
                    self._n_active -= 1
                    if self._n_active == 0:
                        self._run_btn.config(state="normal")
                        self._stop_btn.config(state="disabled")
                        if self._run_best:
                            self._save_btn.config(state="normal")
                        self._status_var.set("Error — see dialog")
                    messagebox.showerror(
                        f"Optimization error (run {run_id + 1})", str(exc))

        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker(self, run_id, ar, N, freq, SNR, p, chain, mirror, combine_in, cb, stage_cb):
        try:
            best_res, best_pos = ar.optimize_chain(
                N, freq, SNR, p, chain,
                mirror=mirror, combine_in=combine_in,
                step_callback=cb,
                stop_event=self._stop_event,
                stage_callback=stage_cb,
            )
            self._result_queue.put(("done", run_id, best_res, best_pos))
        except Exception as exc:
            self._result_queue.put(("error", run_id, exc))

    # ── Run / Stop ────────────────────────────────────────────────────────────

    def _on_run(self):
        # Read parameters
        N    = _int(self._N_entry,    default=3)
        freq = _float(self._freq_entry, default=10.0)
        SNR  = _float(self._SNR_entry,  default=15.0)
        p    = _float(self._p_entry,    default=0.3)
        mirror     = self._mirror_combo.get()
        mode       = self._mode_combo.get()
        if mode == "volume multipleX":
            x_val = max(2, _int(self._multiple_x, default=2))
            mode = f"volume multiple{x_val}"
        combine_in = self._combine_in_var.get()
        method     = self._opt_combo.get()

        try:
            n_runs = max(1, int(self._n_runs_spin.get()))
        except ValueError:
            n_runs = 1

        # Build optimizer chain
        adam_opts = {
            "learning_rate":   _float(self._lr_entry, default=1e-3),
            "max_steps":       _int(self._adam_steps, default=500),
            "tolerance_steps": _int(self._adam_tol_steps, default=50),
        }
        de_opts = {
            "niter":   _int(self._de_niter,   default=200),
            "popsize": _int(self._de_popsize, default=15),
        }
        pso_opts = {
            "niter":      _int(self._pso_niter,      default=100),
            "swarm_size": _int(self._pso_swarm_size, default=400),
        }

        chain_map = {
            "Adam":     [("Adam", adam_opts)],
            "DE":       [("DE",   de_opts)],
            "PSO":      [("PSO",  pso_opts)],
            "DE->Adam": [("DE",   de_opts),  ("Adam", adam_opts)],
            "PSO->Adam":[("PSO",  pso_opts), ("Adam", adam_opts)],
        }
        chain = chain_map.get(method, [("Adam", adam_opts)])

        # Build AnalyticResidual (one instance, shared read-only across threads)
        try:
            extra = {"e1": AO.e1_sym, "e2": AO.e2_sym} if "forcesym" in mode else {}
            ar = AO.AnalyticResidual(default_mode=mode, default_mirror=mirror, **extra)
        except Exception as exc:
            messagebox.showerror("Setup error", str(exc))
            return

        # Initialize per-run bookkeeping
        self._run_histories    = {i: {"steps": [], "residuals": []} for i in range(n_runs)}
        self._run_best         = {}
        self._run_step_offsets = {i: 0 for i in range(n_runs)}
        self._run_stage_lines  = {i: [] for i in range(n_runs)}
        self._n_active      = n_runs
        self._n_runs_total  = n_runs
        self._shared_ar     = ar
        self._shared_N      = N
        self._stop_event.clear()
        self._last_plot_time = 0.0

        self._run_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._save_btn.config(state="disabled")
        self._status_var.set(f"Running {n_runs} parallel run(s)…")

        for run_id in range(n_runs):
            cb       = self._make_callback(run_id)
            stage_cb = self._make_stage_callback(run_id)
            t = threading.Thread(
                target=self._worker,
                args=(run_id, ar, N, freq, SNR, p, chain, mirror, combine_in, cb, stage_cb),
                daemon=True,
            )
            t.start()

    def _on_stop(self):
        self._stop_event.set()
        self._status_var.set("Stopping…")
        self._stop_btn.config(state="disabled")

    def _on_save(self):
        if not self._run_best:
            messagebox.showwarning("Nothing to save", "Run an optimization first.")
            return

        # Ask for a folder name, then a parent directory.
        folder_name = simpledialog.askstring(
            "Save results",
            "Folder name:",
            initialvalue="optimization_results",
            parent=self.root,
        )
        if not folder_name:
            return

        parent = filedialog.askdirectory(
            title="Choose where to create the folder",
            parent=self.root,
        )
        if not parent:
            return

        folder = os.path.join(parent, folder_name)
        os.makedirs(folder, exist_ok=True)

        # Save both matplotlib figures.
        self._fig3d.savefig(
            os.path.join(folder, "positions_3d.png"), dpi=150, bbox_inches="tight")
        self._fig_res.savefig(
            os.path.join(folder, "residual_history.png"), dpi=150, bbox_inches="tight")

        # Write a plain-text summary for every run.
        with open(os.path.join(folder, "results.txt"), "w") as fh:
            fh.write(f"n_runs = {self._n_runs_total}\n")
            fh.write(f"N = {self._shared_N}\n\n")
            for run_id in sorted(self._run_best):
                res, state = self._run_best[run_id]
                fh.write(f"Run {run_id + 1}:\n")
                fh.write(f"  residual = {res}\n")
                fh.write(f"  state    = {np.array(state).tolist()}\n\n")
            best_id = min(self._run_best, key=lambda k: self._run_best[k][0])
            best_res, best_state = self._run_best[best_id]
            fh.write(f"Best run : {best_id + 1}\n")
            fh.write(f"Best residual : {best_res}\n")
            fh.write(f"Best state    : {np.array(best_state).tolist()}\n")

        # Also write the best result in writeFile2 format so ReadData2 can load it.
        if self._shared_ar is not None:
            try:
                ar  = self._shared_ar
                N   = self._shared_N
                freq = _float(self._freq_entry)
                SNR  = _float(self._SNR_entry)
                p    = _float(self._p_entry)
                savepath = os.path.join(folder, "best_result")
                ar.writeFile2(best_res, best_state, N, freq, SNR, p, savepath)
            except Exception:
                pass  # writeFile2 failure is non-fatal

        messagebox.showinfo("Saved", f"Results saved to:\n{folder}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = OptimizationGUI(root)
    root.mainloop()
