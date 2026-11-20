"""Metrics, paired statistics, behavioural signatures and figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

P_MIN = 1e-6
N_BOOT = 10_000
BOOT_SEED = 0
MODEL_ORDER = ["Uniform", "WSLS", "Q-learning", "Sticky Q-learning", "Claude (Haiku)"]
COGNITIVE = ["WSLS", "Q-learning", "Sticky Q-learning"]
LLM = "Claude (Haiku)"
COLORS = {"Human": "#4a4a4a", "Uniform": "#b8b8b8", "WSLS": "#e6a23c",
          "Q-learning": "#5b8fd9", "Sticky Q-learning": "#2c5aa0", "Claude (Haiku)": "#c8553d"}


def trial_metrics(probs: np.ndarray, choice: int) -> dict:
    """NLL (clipped), accuracy, Brier.

    Accuracy is the expected argmax accuracy under random tie-breaking: if the choice
    is among m options tied for the maximum probability, it scores 1/m. (Plain
    np.argmax would always pick the first-listed option on ties, so a uniform
    predictor's accuracy would equal the human base rate of the first-listed option.)
    """
    y = np.zeros_like(probs)
    y[choice] = 1.0
    is_max = np.isclose(probs, probs.max())
    return {
        "nll": float(-np.log(np.clip(probs[choice], P_MIN, 1.0))),
        "accuracy": float(is_max[choice] / is_max.sum()),
        "brier": float(((probs - y) ** 2).sum()),
    }


def participant_summary(pred: pd.DataFrame) -> pd.DataFrame:
    return (pred.groupby(["participant", "model"])[["nll", "accuracy", "brier"]]
            .mean().reset_index())


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def model_summary(psum: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in [m for m in MODEL_ORDER if m in psum.model.unique()]:
        d = psum[psum.model == m]
        lo, hi = bootstrap_mean_ci(d.nll.to_numpy())
        rows.append({"model": m, "mean_nll": d.nll.mean(), "nll_ci_low": lo, "nll_ci_high": hi,
                     "accuracy": d.accuracy.mean(), "brier": d.brier.mean()})
    return pd.DataFrame(rows)


HIGHER_IS_BETTER = {"nll": False, "brier": False, "accuracy": True}


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family = the comparisons passed in)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired_stats(psum: pd.DataFrame, target: str = LLM, metric: str = "nll") -> pd.DataFrame:
    """Per-participant differences in a metric (target - baseline), 95% bootstrap CI,
    two-sided Wilcoxon signed-rank p (Holm-adjusted across baselines), and the number
    of participants for whom the target is strictly better."""
    wide = psum.pivot(index="participant", columns="model", values=metric)
    rows = []
    for b in [m for m in MODEL_ORDER if m in wide.columns and m != target]:
        diff = (wide[target] - wide[b]).to_numpy()
        lo, hi = bootstrap_mean_ci(diff)
        p = wilcoxon(diff, alternative="two-sided").pvalue if np.any(diff != 0) else 1.0
        better = diff > 0 if HIGHER_IS_BETTER[metric] else diff < 0
        rows.append({"metric": metric, "baseline": b, "mean_diff": diff.mean(),
                     "ci_low": lo, "ci_high": hi, "wilcoxon_p": float(p),
                     "n_target_better": int(better.sum()), "n_participants": len(diff)})
    out = pd.DataFrame(rows)
    out.insert(out.columns.get_loc("wilcoxon_p") + 1, "wilcoxon_p_holm", holm(out.wilcoxon_p))
    return out


def win_counts(psum: pd.DataFrame) -> pd.Series:
    wide = psum.pivot(index="participant", columns="model", values="nll")
    best = wide.idxmin(axis=1)
    return best.value_counts().reindex(wide.columns, fill_value=0)


def wsls_signatures(pred: pd.DataFrame) -> pd.DataFrame:
    """Win-stay and lose-shift rates on evaluation trials: empirical for humans, and
    implied by each model's predicted probability of repeating the previous choice.
    Pooled over trials; 95% CI by bootstrap over participants."""
    rows = []
    pred = pred.dropna(subset=["prev_win"])
    base = pred[pred.model == pred.model.iloc[0]]
    sources = [("Human", base, "human_stay")] + [
        (m, pred[pred.model == m], "p_stay") for m in MODEL_ORDER if m in pred.model.unique()]
    for name, d, col in sources:
        for cond, win in [("win-stay", True), ("lose-shift", False)]:
            sub = d[d.prev_win == win]
            vals = sub[col] if win else 1 - sub[col]
            per = vals.groupby(sub.participant).agg(["sum", "count"])
            rng = np.random.default_rng(BOOT_SEED)
            idx = rng.integers(0, len(per), size=(N_BOOT, len(per)))
            s, c = per["sum"].to_numpy(), per["count"].to_numpy()
            boots = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
            rows.append({"source": name, "measure": cond, "rate": vals.mean(),
                         "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                         "n_trials": len(sub)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_model_nll(msum: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(msum))
    err = np.vstack([msum.mean_nll - msum.nll_ci_low, msum.nll_ci_high - msum.mean_nll])
    ax.bar(x, msum.mean_nll, yerr=err, capsize=4, color=[COLORS[m] for m in msum.model],
           edgecolor="none", width=0.65)
    ax.set_xticks(x, msum.model, rotation=15, ha="right")
    ax.set_ylabel("Mean NLL per trial (lower is better)")
    ax.set_title("Out-of-sample NLL by model", fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_participant_scatter(psum: pd.DataFrame, best_baseline: str, path: Path):
    wide = psum.pivot(index="participant", columns="model", values="nll")
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    lim = [0, max(wide[[LLM, best_baseline]].max().max() * 1.08, 0.1)]
    ax.plot(lim, lim, color="#999999", linestyle="--", linewidth=1, label="y = x")
    ax.scatter(wide[best_baseline], wide[LLM], color=COLORS[LLM], s=36, zorder=3,
               edgecolor="white", linewidth=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"{best_baseline}: mean NLL")
    ax.set_ylabel(f"{LLM}: mean NLL")
    ax.set_title("Per-participant NLL", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_wsls(sig: pd.DataFrame, path: Path):
    sources = list(dict.fromkeys(sig.source))
    measures = ["win-stay", "lose-shift"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.8 / len(sources)
    for i, s in enumerate(sources):
        d = sig[sig.source == s].set_index("measure").loc[measures]
        xs = np.arange(len(measures)) + (i - (len(sources) - 1) / 2) * w
        err = np.vstack([d.rate - d.ci_low, d.ci_high - d.rate])
        ax.bar(xs, d.rate, width=w, yerr=err, capsize=2, color=COLORS[s], label=s,
               edgecolor="none")
    ax.set_xticks(range(len(measures)), ["P(stay | win)", "P(shift | loss)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate on evaluation trials")
    ax.set_title("Win-stay and lose-shift rates", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
