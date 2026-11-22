"""Reference checks for the optional RidgeRegression addition."""
import numpy as np
import pytest
from mlforge.ridge_regression import RidgeRegression


def test_alpha_zero_matches_lstsq():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4))
    y = 2.3 + X @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(size=80) * 0.1
    actual = RidgeRegression(alpha=0).fit(X, y)
    reference = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)[0]
    np.testing.assert_allclose(actual.coef_, reference[:-1], atol=1e-10)
    np.testing.assert_allclose(actual.intercept_, reference[-1], atol=1e-10)


def test_regularization_shrinks_coefficients():
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = 3.0 * X[:, 0] + 7.0
    plain = RidgeRegression(alpha=0).fit(X, y)
    regularized = RidgeRegression(alpha=20).fit(X, y)
    assert abs(regularized.coef_[0]) < abs(plain.coef_[0])
    assert regularized.intercept_ == pytest.approx(7.0)


def test_rank_deficiency_gives_finite_solution():
    X = np.array([[1., 2.], [2., 4.], [3., 6.], [4., 8.]])
    model = RidgeRegression(alpha=1.0).fit(X, [1., 3., 5., 7.])
    assert np.isfinite(model.predict(X)).all()


def test_no_intercept():
    X = np.array([1., 2., 3., 4.])
    model = RidgeRegression(alpha=0, fit_intercept=False).fit(X, [2., 4., 6., 8.])
    assert model.intercept_ == 0.0
    np.testing.assert_allclose(model.predict([5., 6.]), [10., 12.], atol=1e-10)


@pytest.mark.parametrize('alpha', [-1, np.nan, np.inf, True, '1'])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        RidgeRegression(alpha=alpha)


def test_unfitted():
    with pytest.raises(RuntimeError):
        RidgeRegression().predict([[1.0]])


def test_feature_count_validation():
    model = RidgeRegression().fit([[1, 2], [3, 4], [5, 6]], [1, 2, 3])
    with pytest.raises(ValueError):
        model.predict([[1]])


def test_score():
    model = RidgeRegression(alpha=0).fit([1, 2, 3, 4], [3, 5, 7, 9])
    assert model.score([1, 2, 3, 4], [3, 5, 7, 9]) == pytest.approx(1.0)
