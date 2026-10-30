"""Produce experimental plots and CSV summaries from saved measurements."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import CONFIGS, ROOT, lr_at, write_json

LABELS = {
    "adam": "Adam (no decay)", "adamw_001": "AdamW, wd=0.01",
    "adamw_01": "AdamW, wd=0.1", "l2_000001": "Adam+L2, wd=1e-5",
    "l2_0001": "Adam+L2, wd=0.001", "l2_01": "Adam+L2, wd=0.1",
}
COLORS = dict(zip(CONFIGS, ["#334155", "#06a6a0", "#0072b2", "#8b5cf6", "#e69f00", "#d55e00"]))
FIG = ROOT / "results" / "figures"


def savefig(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def quadratic():
    a = np.array([[4., 1.], [1., 2.]])
    b = np.array([3., -1.])
    d = np.diag([0.05, 1.])
    ds = np.diag(np.sqrt(np.diag(d)))
    lam, eta = 0.3, 0.4
    xw = np.linalg.solve(a + lam * np.linalg.inv(d), b)
    xl = np.linalg.solve(a + lam * np.eye(2), b)
    xf = np.linalg.solve(a, b)
    curv = ds @ a @ ds + lam * np.eye(2)
    mu, smooth = np.linalg.eigvalsh(curv)[[0, -1]]
    x, y = np.array([-1.5, 1.5]), np.array([-1.5, 1.5])
    xs, ys, gaps = [], [], []
    energy = lambda z: 0.5 * z @ a @ z - b @ z + 0.5 * lam * z @ np.linalg.solve(d, z)
    for i in range(201):
        xs.append(x.copy())
        ys.append(y.copy())
        gaps.append(max(0., energy(x) - energy(xw)))
        x -= eta * (d @ (a @ x - b) + lam * x)
        y -= eta * d @ (a @ y - b + lam * y)
    xs, ys, gaps = np.array(xs), np.array(ys), np.array(gaps)
    bound = gaps[0] * (1 - eta * mu) ** np.arange(201)
    # Checks the closed-form minimizer, convergence, and theoretical energy bound.
    assert np.linalg.norm(xs[-1] - xw) < 1e-8
    assert np.linalg.norm(ys[-1] - xl) < 1e-4
    assert np.all(gaps <= bound + 1e-12)
    xx, yy = np.meshgrid(np.linspace(-1.7, 1.6, 200), np.linspace(-1.5, 1.7, 200))
    zz = 0.5 * (4 * xx**2 + 2 * xx * yy + 2 * yy**2) - 3 * xx + yy
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].contour(xx, yy, zz, levels=18, colors="#cbd5e1", linewidths=0.6)
    axs[0].plot(xs[:, 0], xs[:, 1], "o-", ms=2, color="#0072b2", label="Decoupled (frozen D)")
    axs[0].plot(ys[:, 0], ys[:, 1], "s-", ms=2, color="#d55e00", label="Coupled (frozen D)")
    axs[0].scatter(*xf, marker="*", s=110, color="black", label="Unregularized optimum")
    axs[0].scatter(*xw, marker="x", s=80, color="#0072b2")
    axs[0].scatter(*xl, marker="x", s=80, color="#d55e00")
    axs[0].set(xlabel="$x_1$", ylabel="$x_2$", title="A. Different stationary points")
    axs[0].legend(fontsize=8)
    ts = np.arange(61)
    axs[1].semilogy(ts, np.maximum(gaps[:61], 1e-15), label="Measured objective gap", color="#0072b2")
    axs[1].semilogy(ts, bound[:61], "--", label="Proved upper bound", color="#334155")
    axs[1].set(xlabel="Iteration", ylabel="$F_D(x_t)-F_D(x_*)$", title="B. Linear convergence")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "quadratic")
    rows = [dict(t=i, x1=xs[i, 0], x2=xs[i, 1], coupled_x1=ys[i, 0], coupled_x2=ys[i, 1],
                 objective_gap=gaps[i], upper_bound=bound[i]) for i in range(201)]
    with (ROOT / "results" / "quadratic.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(A=a.tolist(), b=b.tolist(), D=d.tolist(), weight_decay=lam, eta=eta,
                  x_decoupled=xw.tolist(), x_coupled=xl.tolist(), x_unregularized=xf.tolist(),
                  mu=mu, smooth=smooth, contraction_bound=1 - eta * mu,
                  final_error=float(np.linalg.norm(xs[-1] - xw)))
    write_json(ROOT / "results" / "quadratic.json", result)
    return result


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2, "pdf.fonttype": 42})
    q = quadratic()
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "results" / "main").glob("*/summary.json"))]
    assert len(summaries) == 3 * len(CONFIGS), f"Expected {3 * len(CONFIGS)} completed runs, got {len(summaries)}"
    for seed in [11, 22, 33]:
        group = [r for r in summaries if r["seed"] == seed]
        assert len({r["initial_hash"] for r in group}) == 1
        assert all(r["steps"] == 1000 and r["batch"] == 32 for r in group)
    fields = ["config", "seed", "peak_lr", "weight_decay", "train_ce", "val_ce", "test_ce", "test_ppl",
              "param_norm", "train_seconds", "wall_total_seconds", "tokens_per_second", "peak_cuda_mb"]
    with (ROOT / "results" / "runs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    stats = {}
    for name in CONFIGS:
        rows = [r for r in summaries if r["config"] == name]
        stats[name] = {k: dict(mean=float(np.mean([r[k] for r in rows])), sd=float(np.std([r[k] for r in rows], ddof=1)))
                       for k in fields[4:]}
    pilots = [json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "results" / "pilot").glob("*/summary.json")]
    with (ROOT / "results" / "pilot_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "peak_lr", "val_ce"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(pilots, key=lambda r: (r["config"], r["peak_lr"])))
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    all_curves = {}
    for name in CONFIGS:
        curves = []
        for row in sorted((r for r in summaries if r["config"] == name), key=lambda r: r["seed"]):
            folder = ROOT / "results" / "main" / f"{name}_seed{row['seed']}_lr0.001_steps1000"
            curves.append([json.loads(s) for s in (folder / "metrics.jsonl").read_text().splitlines()])
        steps = np.array([r["step"] for r in curves[0]])
        all_curves[name] = curves
        for ax, key in zip(axs, ["train_ce", "val_ce", "param_norm"]):
            array = np.array([[r[key] for r in c] for c in curves])
            mean, sd = array.mean(axis=0), array.std(axis=0, ddof=1)
            style = "--" if name in ["adamw_001", "l2_000001"] else "-"
            ax.plot(steps, mean, style, color=COLORS[name], label=LABELS[name], lw=1.6)
            ax.fill_between(steps, mean - sd, mean + sd, color=COLORS[name], alpha=0.1)
    for ax, title, ylabel in zip(axs, ["A. Training loss", "B. Validation loss", "C. Parameter norm"],
                               ["Cross entropy (nats/char)", "Cross entropy (nats/char)", "$\\|\\theta\\|_2$"]):
        ax.set(title=title, xlabel="Optimizer steps", ylabel=ylabel)
    for ax in axs[:2]:
        ax.set_ylim(1.7, 3.0)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    savefig(fig, "training")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    xs = np.arange(len(CONFIGS))
    short = ["Adam\n0", "AdamW\n0.01", "AdamW\n0.1", "L2\n1e-5", "L2\n0.001", "L2\n0.1"]
    for i, name in enumerate(CONFIGS):
        rows = [r for r in summaries if r["config"] == name]
        axs[0].errorbar(i, stats[name]["test_ce"]["mean"], yerr=stats[name]["test_ce"]["sd"],
                       fmt="o", capsize=4, color=COLORS[name], ms=6)
        axs[0].scatter(i + np.array([-0.12, 0, 0.12]), [r["test_ce"] for r in rows],
                       s=14, alpha=0.6, color=COLORS[name])
    axs[0].set(xticks=xs, xticklabels=short, ylabel="Test CE (nats/char)", title="A. Final test results (3 seeds)")
    base = {r["seed"]: r for r in summaries if r["config"] == "adam"}
    paired = {}
    for i, name in enumerate(list(CONFIGS)[1:]):
        rows = sorted((r for r in summaries if r["config"] == name), key=lambda r: r["seed"])
        diffs = np.array([r["test_ce"] - base[r["seed"]]["test_ce"] for r in rows])
        paired[name] = dict(seeds=[r["seed"] for r in rows], differences=diffs.tolist(),
                            mean=float(diffs.mean()), sd=float(diffs.std(ddof=1)))
        axs[1].scatter([i] * 3, diffs, color=COLORS[name], s=25)
    axs[1].axhline(0, color="black", lw=0.8)
    axs[1].set(xticks=np.arange(len(CONFIGS) - 1), xticklabels=short[1:],
               ylabel="Test CE minus paired Adam CE", title="B. Paired differences (lower is better)")
    fig.tight_layout()
    savefig(fig, "test_results")
    decay_product = math.prod(1 - lr_at(t, 1000, 1e-3) * 0.1 for t in range(1000))
    total_train = sum(r["train_seconds"] for r in summaries + pilots)
    overview = dict(stats=stats, paired=paired, quadratic=q, decay_product_wd01=decay_product,
                    all_training_seconds=total_train, main_runs=len(summaries), pilot_runs=len(pilots),
                    total_main_tokens=sum(r["training_tokens"] for r in summaries),
                    total_pilot_tokens=sum(r["training_tokens"] for r in pilots),
                    dataset=json.loads((ROOT / "results" / "dataset.json").read_text(encoding="utf-8")))
    write_json(ROOT / "results" / "aggregate.json", overview)
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
