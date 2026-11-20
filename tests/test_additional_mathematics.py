"""Mathematical checks through the unchanged original public estimator APIs."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from mlforge import LinearRegression, LogisticRegression

@pytest.mark.parametrize("fit_intercept", [False, True])
def test_least_squares_residual_is_orthogonal_to_design(fit_intercept):
    rng = np.random.default_rng(438)
    X = rng.normal(size=(40, 3))
    y = 1.5 + X @ np.array([2., -1., .5]) + rng.normal(0., .2, 40)
    model = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    A = np.column_stack((np.ones(len(X)), X)) if fit_intercept else X
    residual = y - model.predict(X)
    assert_allclose(A.T @ residual, 0., atol=1e-11)


def test_full_rank_linear_predictions_are_invariant_to_affine_feature_scaling():
    rng = np.random.default_rng(382)
    X = rng.normal(size=(35, 3))
    query = rng.normal(size=(7, 3))
    y = X @ np.array([2., -3., .4]) + rng.normal(0., .1, 35)
    offset, scale = np.array([5., -2., 8.]), np.array([2., .5, 3.])
    raw = LinearRegression().fit(X, y)
    transformed = LinearRegression().fit((X-offset)/scale, y)
    assert_allclose(raw.predict(query), transformed.predict((query-offset)/scale), atol=1e-11)


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("seed", [11, 42, 438])
def test_public_logistic_update_matches_central_finite_differences(fit_intercept, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(9, 3))
    y = np.array([0., 1., 0., 1., 1., 0., 0., 1., 0.])
    A = np.column_stack((np.ones(len(X)), X)) if fit_intercept else X
    eta = .2
    first = LogisticRegression(learning_rate=eta, max_iter=1, fit_intercept=fit_intercept).fit(X, y)
    second = LogisticRegression(learning_rate=eta, max_iter=2, fit_intercept=fit_intercept).fit(X, y)
    theta = np.r_[first.intercept_, first.coef_] if fit_intercept else first.coef_
    next_theta = np.r_[second.intercept_, second.coef_] if fit_intercept else second.coef_
    analytic = (theta-next_theta)/eta

    def objective(parameters):
        z = A @ parameters
        # Independent objective expression on moderate logits.
        return np.mean(np.logaddexp(0., z) - y*z)

    h = 1e-5
    numeric = np.empty_like(theta)
    for j in range(len(theta)):
        shift = np.zeros_like(theta)
        shift[j] = h
        numeric[j] = (objective(theta+shift)-objective(theta-shift))/(2*h)
    assert_allclose(analytic, numeric, atol=1e-9, rtol=1e-6)


def test_two_logistic_updates_match_hand_constructed_recurrence():
    X = np.array([[1., -2.], [0., 1.], [-1., .5], [2., 1.]])
    y = np.array([0., 1., 0., 1.])
    A = np.column_stack((np.ones(len(X)), X))
    theta = np.zeros(3)
    for _ in range(2):
        p = 1. / (1. + np.exp(-(A @ theta)))
        theta -= .2 * np.mean((p-y)[:, None] * A, axis=0)
    model = LogisticRegression(learning_rate=.2, max_iter=2).fit(X, y)
    assert_allclose(np.r_[model.intercept_, model.coef_], theta, atol=1e-12)
