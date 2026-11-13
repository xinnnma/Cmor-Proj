# Unit tests

`test_linear_regression.py` checks the implemented NumPy model against
analytical solutions. Coverage includes predictions, multiple features,
optional intercepts, noisy least squares, R², invalid inputs, unfitted models,
rank-deficient and underdetermined designs, and refitting.

`test_logistic_regression.py` checks binary training, predictions, probability
shapes and class ordering, accuracy, optional intercepts, label and parameter
validation, numerical stability, refitting, and convergence to a known
nonseparable solution. Datasets are deterministic; numerical assertions use
tolerances. The existing Linear Regression tests remain unchanged.

`test_reference_consistency.py` adds four checks using deterministic
synthetic data. For Linear Regression, it compares coefficients and
predictions with a NumPy least-squares reference on correlated features,
and checks predictions when features are exactly linearly dependent.
For Logistic Regression, it checks that training loss remains finite
and decreases overall, and that predicted probabilities are finite,
lie between zero and one, and sum to one for each sample.

Name test files `test_*.py` and import the installed `mlforge` package. From the
project root, run `uv run pytest -v`.

Numerical assertions use floating-point tolerances. Tests import the installed
package and do not require scikit-learn. Add real tests alongside future
algorithms; do not add placeholder passing tests.
