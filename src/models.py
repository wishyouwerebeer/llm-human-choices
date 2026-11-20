"""Classic cognitive baseline models for a K-armed bandit.

Every model maps a full choice/reward sequence to a (T, K) matrix whose row t is the
predicted distribution over options for trial t, using only trials 0..t-1. Parameters
are fitted by maximum likelihood on the first half; evaluation rows are read from the
same forward pass, so learned values (Q, previous choice) carry into the second half.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

REWARD_SCALE = 100.0   # points are ~1-100; scale to [0, 1]
Q0 = 0.5               # initial Q-value (mid-range of scaled rewards)
P_MIN = 1e-6


def is_win(rewards: np.ndarray, t: int) -> bool | None:
    """Outcome of trial t is a 'win' if its reward >= the previous trial's reward.

    Undefined (None) for t == 0. In a restless bandit, raw points carry no fixed
    win/loss meaning, so we use a relative (improvement) criterion.
    """
    if t <= 0:
        return None
    return bool(rewards[t] >= rewards[t - 1])


# ---------------------------------------------------------------- forward passes
def uniform_probs(choices: np.ndarray, rewards: np.ndarray, k: int) -> np.ndarray:
    return np.full((len(choices), k), 1.0 / k)


def wsls_probs(params, choices: np.ndarray, rewards: np.ndarray, k: int) -> np.ndarray:
    """Win-stay / lose-shift with lapse epsilon:
    p = (1 - eps) * deterministic_rule + eps * uniform.
    After a loss, the shift probability is spread evenly over the other K-1 options.
    Trials 0 and 1 (no defined previous outcome) are predicted uniformly.
    """
    (eps,) = params
    T = len(choices)
    P = np.full((T, k), 1.0 / k)
    for t in range(2, T):
        prev = choices[t - 1]
        rule = np.zeros(k)
        if is_win(rewards, t - 1):
            rule[prev] = 1.0
        else:
            rule[:] = 1.0 / (k - 1)
            rule[prev] = 0.0
        P[t] = (1 - eps) * rule + eps / k
    return P


def q_probs(params, choices: np.ndarray, rewards: np.ndarray, k: int,
            sticky: bool = False) -> np.ndarray:
    """Delta-rule Q-learning with softmax choice; optional perseveration bonus kappa
    added to the previously chosen option's logit."""
    if sticky:
        alpha, beta, kappa = params
    else:
        (alpha, beta), kappa = params, 0.0
    T = len(choices)
    P = np.empty((T, k))
    q = np.full(k, Q0)
    r = rewards / REWARD_SCALE
    prev = -1
    for t in range(T):
        logits = beta * q
        if prev >= 0:
            logits[prev] += kappa
        logits -= logits.max()
        e = np.exp(logits)
        P[t] = e / e.sum()
        c = choices[t]
        q[c] += alpha * (r[t] - q[c])
        prev = c
    return P


# ---------------------------------------------------------------- fitting
def nll(P: np.ndarray, choices: np.ndarray) -> float:
    p = np.clip(P[np.arange(len(choices)), choices], P_MIN, 1.0)
    return float(-np.log(p).sum())


@dataclass
class ModelSpec:
    name: str
    bounds: list[tuple[float, float]]
    forward: Callable[..., np.ndarray]
    param_names: list[str] = field(default_factory=list)


MODELS: dict[str, ModelSpec] = {
    "WSLS": ModelSpec("WSLS", [(0.0, 1.0)], wsls_probs, ["epsilon"]),
    "Q-learning": ModelSpec("Q-learning", [(0.0, 1.0), (0.0, 50.0)],
                            lambda th, c, r, k: q_probs(th, c, r, k, sticky=False),
                            ["alpha", "beta"]),
    "Sticky Q-learning": ModelSpec("Sticky Q-learning", [(0.0, 1.0), (0.0, 50.0), (-5.0, 5.0)],
                                   lambda th, c, r, k: q_probs(th, c, r, k, sticky=True),
                                   ["alpha", "beta", "kappa"]),
}


def fit_model(spec: ModelSpec, choices: np.ndarray, rewards: np.ndarray, k: int,
              fit_end: int, n_restarts: int = 10, seed: int = 42) -> tuple[np.ndarray, float]:
    """MLE on trials [0, fit_end) with L-BFGS-B from n_restarts uniform random starts.

    The forward pass is run on the fitting trials only, so no second-half data
    influences the parameters.
    """
    c_fit, r_fit = choices[:fit_end], rewards[:fit_end]
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in spec.bounds])
    hi = np.array([b[1] for b in spec.bounds])
    obj = lambda th: nll(spec.forward(th, c_fit, r_fit, k), c_fit)
    best_x, best_f = None, np.inf
    for _ in range(n_restarts):
        x0 = rng.uniform(lo, hi)
        res = minimize(obj, x0, method="L-BFGS-B", bounds=spec.bounds)
        if res.fun < best_f:
            best_x, best_f = res.x, float(res.fun)
    return best_x, best_f
