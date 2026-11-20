import numpy as np

from src.analysis import holm, trial_metrics


def test_uniform_accuracy_is_chance_regardless_of_choice():
    for c in range(4):
        assert np.isclose(trial_metrics(np.full(4, 0.25), c)["accuracy"], 0.25)


def test_accuracy_ties_and_clear_argmax():
    p = np.array([0.4, 0.4, 0.1, 0.1])
    assert trial_metrics(p, 0)["accuracy"] == 0.5
    assert trial_metrics(p, 2)["accuracy"] == 0.0
    assert trial_metrics(np.array([0.1, 0.7, 0.1, 0.1]), 1)["accuracy"] == 1.0


def test_nll_and_brier():
    m = trial_metrics(np.full(4, 0.25), 2)
    assert np.isclose(m["nll"], np.log(4))
    assert np.isclose(m["brier"], 0.75)
    assert np.isclose(trial_metrics(np.array([1.0, 0, 0, 0]), 1)["nll"], -np.log(1e-6))


def test_holm():
    assert np.allclose(holm([0.0012, 0.0049, 0.040, 0.261]), [0.0048, 0.0147, 0.080, 0.261])
    assert np.allclose(holm([0.04, 0.01]), [0.04, 0.02])
