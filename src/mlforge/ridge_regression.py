"""L2-regularized least squares implemented with NumPy only."""
from __future__ import annotations

from numbers import Real
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linear_regression import _validate_X, _validate_y


class RidgeRegression:
    """Fit squared-error regression with an L2 penalty on coefficients.

    Minimizes ||y - X @ w - b||_2**2 + alpha * ||w||_2**2.
    The intercept, when enabled, is never penalized.

    Parameters
    ----------
    alpha : float, default=1.0
        Nonnegative L2 regularization strength. alpha=0 recovers least squares.
    fit_intercept : bool, default=True
        Whether to fit an unpenalized intercept.
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite, nonnegative real number.")
        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be a boolean.")
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> RidgeRegression:
        X_array = _validate_X(X)
        y_array = _validate_y(y, X_array.shape[0])
        if self.fit_intercept:
            x_mean = X_array.mean(axis=0)
            y_mean = float(y_array.mean())
            X_centered = X_array - x_mean
            y_centered = y_array - y_mean
        else:
            x_mean = np.zeros(X_array.shape[1], dtype=float)
            y_mean = 0.0
            X_centered = X_array
            y_centered = y_array
        # SVD avoids squaring the condition number through X.T @ X.
        # For alpha=0, zero singular values map to zero, yielding the
        # minimum-norm least-squares solution (like numpy.linalg.lstsq).
        try:
            U, singular, Vt = np.linalg.svd(X_centered, full_matrices=False)
            if self.alpha == 0.0:
                cutoff = np.finfo(float).eps * max(X_centered.shape) * singular[0]
                weights = np.divide(1.0, singular, out=np.zeros_like(singular), where=singular > cutoff)
            else:
                weights = singular / (singular ** 2 + self.alpha)
            coefficients = Vt.T @ (weights * (U.T @ y_centered))
            intercept = y_mean - float(x_mean @ coefficients) if self.fit_intercept else 0.0
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            raise ValueError("Numerical fitting failed; consider scaling the input features.") from exc
        if not np.all(np.isfinite(coefficients)) or not np.isfinite(intercept):
            raise ValueError("Numerical overflow during fitting; scale the input features.")
        self.coef_ = coefficients
        self.intercept_ = intercept
        self.n_features_in_ = X_array.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("RidgeRegression must be fitted before prediction.")
        X_array = _validate_X(X)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_array.shape[1]}.")
        with np.errstate(over="ignore", invalid="ignore"):
            predictions = X_array @ self.coef_ + self.intercept_
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Numerical overflow during prediction; scale the input features.")
        return predictions

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        predictions = self.predict(X)
        target = _validate_y(y, predictions.shape[0])
        if len(target) < 2:
            raise ValueError("R² is undefined with fewer than two samples.")
        total = float(np.sum((target - target.mean()) ** 2))
        if total == 0.0:
            raise ValueError("R² is undefined for a constant target.")
        return 1.0 - float(np.sum((target - predictions) ** 2)) / total
