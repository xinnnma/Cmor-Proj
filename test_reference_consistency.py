"""Independent reference checks for the public MLForge API.

These complement existing edge-case tests by comparing against mathematical
reference implementations on deterministic synthetic data.
"""

import numpy as np

from mlforge import LinearRegression, LogisticRegression


def test_linear_regression_agrees_with_lstsq_on_correlated_features():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(120, 3))
    X[:, 2] = X[:, 0] + 0.05 * X[:, 2]
    y = 1.7 + X @ np.array([2.0, -0.5, 0.8]) + 0.1 * rng.normal(size=120)

    model = LinearRegression().fit(X, y)
    reference = np.linalg.lstsq(
        np.column_stack([np.ones(len(X)), X]), y, rcond=None
    )[0]

    np.testing.assert_allclose(model.intercept_, reference[0], atol=1e-10)
    np.testing.assert_allclose(model.coef_, reference[1:], atol=1e-10)
    np.testing.assert_allclose(model.predict(X), reference[0] + X @ reference[1:], atol=1e-10)


def test_linear_regression_handles_exact_rank_deficiency():
    x = np.linspace(-3.0, 3.0, 40)
    X = np.column_stack([x, 2.0 * x])
    y = 4.0 + 3.0 * x

    model = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-10)


def test_logistic_regression_improves_cross_entropy_on_scaled_data():
    rng = np.random.default_rng(438)
    X = rng.normal(size=(200, 2))
    logits = 1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.15
    y = (rng.random(200) < 1.0 / (1.0 + np.exp(-logits))).astype(int)

    model = LogisticRegression(learning_rate=0.15, max_iter=400).fit(X, y)
    assert len(model.loss_history_) == 401
    assert np.all(np.isfinite(model.loss_history_))
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_logistic_regression_probabilities_are_valid():
    rng = np.random.default_rng(577)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(int)
    model = LogisticRegression(learning_rate=0.1, max_iter=300).fit(X, y)
    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
