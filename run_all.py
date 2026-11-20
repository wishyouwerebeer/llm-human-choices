"""Reproduce the full pipeline: data -> baselines -> Claude predictions -> analysis -> figures.

    python run_all.py              # full run (20 participants; Claude responses cached)
    python run_all.py --pilot      # first 2 sampled participants, Claude only, report call stats
    python run_all.py --workers 1  # sequential Claude calls (default: 2 in parallel)
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import analysis as A
from src.claude_predictor import ClaudePredictor, UsageLimitReached
from src.data import N_EVAL, sample_participants
from src.models import MODELS, fit_model, is_win, uniform_probs

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def run_claude(participants, workers: int):
    predictor = ClaudePredictor()
    jobs = [(p, t) for p in participants for t in p.eval_indices]
    out = {}

    def task(job):
        p, t = job
        probs, rec = predictor.predict(p.data, t)
        return (p.pid, t), probs, rec

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, (key, probs, rec) in enumerate(ex.map(task, jobs), 1):
                out[key] = (probs, rec)
                if i % 10 == 0 or i == len(jobs):
                    print(f"  Claude predictions: {i}/{len(jobs)}", flush=True)
    except UsageLimitReached as e:
        print("\nUsage limit reached; progress is saved in .cache/claude/. "
              "Re-run the same command after the limit resets to resume.\n"
              f"CLI message: {e}", file=sys.stderr)
        sys.exit(2)
    return out, predictor


def predictions_table(participants, claude_out):
    rows, params = [], []
    for p in participants:
        c = np.array([p.data.options.index(x) for x in p.data.choices])
        r = np.array(p.data.rewards)
        k = len(p.data.options)
        prob_mats = {"Uniform": uniform_probs(c, r, k)}
        for name, spec in MODELS.items():
            theta, fit_nll = fit_model(spec, c, r, k, fit_end=p.split)
            prob_mats[name] = spec.forward(theta, c, r, k)
            params.append({"participant": p.pid, "model": name, "fit_nll_sum": fit_nll,
                           "n_fit_trials": p.split,
                           **dict(zip(spec.param_names, map(float, theta)))})
        if claude_out is not None:
            prob_mats[A.LLM] = np.full((len(c), k), np.nan)
            for t in p.eval_indices:
                prob_mats[A.LLM][t] = claude_out[(p.pid, t)][0]
        for t in p.eval_indices:
            for name, P in prob_mats.items():
                probs = P[t]
                rows.append({
                    "participant": p.pid, "trial": t + 1, "model": name,
                    "choice": p.data.choices[t], "prev_choice": p.data.choices[t - 1],
                    "prev_win": is_win(r, t - 1), "human_stay": float(c[t] == c[t - 1]),
                    "p_choice": probs[c[t]], "p_stay": probs[c[t - 1]],
                    "probs": json.dumps(dict(zip(p.data.options, np.round(probs, 6).tolist()))),
                    **A.trial_metrics(probs, c[t]),
                })
    return pd.DataFrame(rows), pd.DataFrame(params)


def claude_report(predictor, claude_out, n_participants):
    recs = [rec for _, rec in claude_out.values()]
    secs = [a["wall_seconds"] for rec in recs for a in rec["attempts"]]
    usage = [a["usage"] or {} for rec in recs for a in rec["attempts"]]
    tok_in = sum(u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                 + u.get("cache_read_input_tokens", 0) for u in usage)
    tok_out = sum(u.get("output_tokens", 0) for u in usage)
    models = sorted({m for rec in recs for m in rec["reported_models"]})
    return {
        "model_alias": predictor.model, "reported_models": models,
        "n_participants": n_participants, "n_predictions": len(recs),
        "n_cli_calls_total": len(secs),
        "n_cli_calls_this_run": predictor.stats["calls"],
        "cache_hits_this_run": predictor.stats["cache_hits"],
        "n_malformed_retries": sum(len(rec["attempts"]) - 1 for rec in recs),
        "n_parse_failures": sum(rec["parse_failure"] for rec in recs),
        "mean_seconds_per_call": float(np.mean(secs)) if secs else None,
        "total_input_tokens": tok_in, "total_output_tokens": tok_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="2 participants, report Claude call stats")
    ap.add_argument("--workers", type=int, default=2, choices=[1, 2])
    args = ap.parse_args()

    participants = sample_participants()
    if args.pilot:
        participants = participants[:2]
    print(f"{len(participants)} participants: {[p.pid for p in participants]}; "
          f"{N_EVAL} evaluation trials each")

    claude_out, predictor = run_claude(participants, args.workers)
    report = claude_report(predictor, claude_out, len(participants))
    if args.pilot:
        print(json.dumps(report, indent=2))
        (RESULTS / "pilot").mkdir(parents=True, exist_ok=True)
        (RESULTS / "pilot" / "pilot_report.json").write_text(json.dumps(report, indent=2))
        return

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    pred, params = predictions_table(participants, claude_out)
    psum = A.participant_summary(pred)
    msum = A.model_summary(psum)
    stats = pd.concat([A.paired_stats(psum, metric=m) for m in ["nll", "brier", "accuracy"]],
                      ignore_index=True)
    wins = A.win_counts(psum)
    sig = A.wsls_signatures(pred)

    pred.to_csv(RESULTS / "predictions.csv", index=False)
    psum.to_csv(RESULTS / "participant_summary.csv", index=False)
    msum.to_csv(RESULTS / "model_summary.csv", index=False)
    stats.to_csv(RESULTS / "paired_stats.csv", index=False)
    params.to_csv(RESULTS / "fitted_parameters.csv", index=False)
    sig.to_csv(RESULTS / "wsls_signatures.csv", index=False)
    wins.rename("n_best").to_csv(RESULTS / "win_counts.csv")
    (RESULTS / "claude_run_report.json").write_text(json.dumps(report, indent=2))

    best = msum[msum.model.isin(A.COGNITIVE)].sort_values("mean_nll").model.iloc[0]
    A.fig_model_nll(msum, FIGURES / "fig1_model_nll.png")
    A.fig_participant_scatter(psum, best, FIGURES / "fig2_participant_scatter.png")
    A.fig_wsls(sig, FIGURES / "fig3_wsls_behavior.png")

    pd.set_option("display.width", 140)
    print("\nModel summary (evaluation trials):\n", msum.round(4).to_string(index=False))
    print("\nPaired differences, Claude - baseline (NLL/Brier: positive = Claude worse; "
          "accuracy: positive = Claude better):\n", stats.round(4).to_string(index=False))
    print("\nWin counts (participants best predicted):\n", wins.to_string())
    print("\nWin-stay / lose-shift:\n", sig.round(3).to_string(index=False))
    print("\nClaude run report:\n", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
