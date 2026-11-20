"""Sanity and parameter-recovery tests for the cognitive baselines."""
import numpy as np
import pytest

from src.models import MODELS, Q0, REWARD_SCALE, fit_model, is_win, q_probs, wsls_probs

K = 4


def simulate_q(alpha, beta, kappa, T=150, seed=0):
    """Simulate a sticky Q-learner on a 4-armed Gaussian random-walk bandit."""
    rng = np.random.default_rng(seed)
    means = rng.uniform(20, 80, K)
    q, prev = np.full(K, Q0), -1
    choices, rewards = [], []
    for _ in range(T):
        logits = beta * q
        if prev >= 0:
            logits[prev] += kappa
        p = np.exp(logits - logits.max()); p /= p.sum()
        c = rng.choice(K, p=p)
        r = float(np.clip(np.round(rng.normal(means[c], 4)), 1, 100))
        means = np.clip(means + rng.normal(0, 2.8, K), 1, 100)
        q[c] += alpha * (r / REWARD_SCALE - q[c])
        choices.append(c); rewards.append(r); prev = c
    return np.array(choices), np.array(rewards)


@pytest.mark.parametrize("name", list(MODELS))
def test_rows_are_distributions(name):
    c, r = simulate_q(0.5, 10, 1.0)
    spec = MODELS[name]
    theta = np.array([(lo + hi) / 2 for lo, hi in spec.bounds])
    P = spec.forward(theta, c, r, K)
    assert P.shape == (len(c), K)
    assert np.all(P >= 0) and np.allclose(P.sum(1), 1.0)


def test_predictions_use_only_past_trials():
    """Changing trial t's choice/reward must not change predictions for trials <= t."""
    c, r = simulate_q(0.5, 10, 1.0)
    c2, r2 = c.copy(), r.copy()
    c2[80] = (c2[80] + 1) % K
    r2[80] = 1.0
    for spec in MODELS.values():
        theta = np.array([(lo + hi) / 2 for lo, hi in spec.bounds])
        P1, P2 = spec.forward(theta, c, r, K), spec.forward(theta, c2, r2, K)
        assert np.allclose(P1[:81], P2[:81])
        assert not np.allclose(P1[81:], P2[81:])


def test_wsls_rule():
    c = np.array([0, 0, 1, 1])
    r = np.array([50.0, 60.0, 40.0, 40.0])
    P = wsls_probs([0.0], c, r, K)
    assert np.allclose(P[2], [1, 0, 0, 0])            # trial 1 was a win -> stay on 0
    assert np.allclose(P[3], [1 / 3, 0, 1 / 3, 1 / 3])  # trial 2 was a loss -> shift from 1
    assert is_win(r, 0) is None and is_win(r, 3) is True


def test_q_learning_parameter_recovery():
    c, r = simulate_q(alpha=0.6, beta=15.0, kappa=0.0, T=600, seed=1)
    theta, _ = fit_model(MODELS["Q-learning"], c, r, K, fit_end=len(c))
    assert abs(theta[0] - 0.6) < 0.25
    assert abs(theta[1] - 15.0) < 6.0


def test_sticky_recovers_positive_kappa():
    c, r = simulate_q(alpha=0.5, beta=10.0, kappa=2.0, T=600, seed=2)
    theta, _ = fit_model(MODELS["Sticky Q-learning"], c, r, K, fit_end=len(c))
    assert theta[2] > 1.0


def test_fit_uses_only_first_half():
    c, r = simulate_q(0.5, 10, 1.0)
    c2 = c.copy(); c2[100:] = 0
    t1, f1 = fit_model(MODELS["Q-learning"], c, r, K, fit_end=75)
    t2, f2 = fit_model(MODELS["Q-learning"], c2, r, K, fit_end=75)
    assert np.allclose(t1, t2) and f1 == f2


def test_sticky_nests_q_learning():
    c, r = simulate_q(0.5, 10, 0.0)
    assert np.allclose(q_probs([0.3, 8.0], c, r, K), q_probs([0.3, 8.0, 0.0], c, r, K, sticky=True))
