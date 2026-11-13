# MLForge — A Machine Learning Package Built from Scratch

## Final project audit

Integration audit: **November 13, 2026**, based on published `main` commit
`0e3e7096218b985c59c52420caa3a7e644d3bf6e` and the local corrections on
`final-integration`. This supersedes the October 20 audit's counts and
presentation requirement. The final deliverable is the GitHub repository;
**no presentation is required**, per the team's instructor update.
Historical presentation notes remain optional reference material.

The [integration report](FINAL_INTEGRATION.md) records teammate commit evidence,
full metrics, GitHub Actions status, and the changes awaiting publication
approval. This audit maps the supplied rubric to verified evidence; it does
not predict an awarded grade.

## Required deliverables

| Requirement | Status | Repository evidence |
| --- | --- | --- |
| Multiple Jupyter notebooks demonstrating machine learning | Complete | Four independent, executed notebooks in [`notebooks/`](../notebooks/README.md), with ten saved figures. |
| Exploration, preprocessing, modeling, and evaluation | Complete | Training-data exploration, training-only standardization, held-out regression and classification evaluation, baselines, and five-fold training-partition cross-validation. |
| Custom Python package with at least two reusable algorithms | Complete | NumPy [`LinearRegression`](../src/mlforge/linear_regression.py) and [`LogisticRegression`](../src/mlforge/logistic_regression.py), exported from `mlforge`. |
| Installation and package usage documentation | Complete | [`README.md`](../README.md) includes locked setup, both executable examples, model APIs, mathematical descriptions, and limitations. |
| Comprehensive pytest tests | Complete | **156 passing cases**: 57 Linear Regression, 95 Logistic Regression, and four reference checks in [`tests/`](../tests/README.md). |
| Practical notebook use of the package | Complete | Notebooks 02–04 train installed MLForge estimators; scikit-learn supplies datasets and utilities. |
| Public GitHub repository and preserved contributions | Complete; local corrections await approval | [xinnnma/Cmor-Proj](https://github.com/xinnnma/Cmor-Proj), public, default branch `main`; all five members' real contributions remain in history and the working tree. |
| Automated testing | Complete | [Workflow](../.github/workflows/ci.yml) checks both imports and runs the full test suite with the locked environment on Ubuntu/Python 3.12. |

## Rubric review

| Category | Weight | Verified evidence and practical limits |
| --- | --- | --- |
| Functionality and Implementation | 40% | Two working NumPy estimators with optional intercepts, prediction, scoring, input validation, and meaningful unfitted errors. Least squares uses `numpy.linalg.lstsq`; binary Logistic Regression uses explicit batch gradient descent with stable sigmoid and loss calculations. Model code and behavior were preserved. |
| Documentation and Readability | 20% | Mathematical docstrings, runnable README examples, four notebook descriptions, verified metrics, accurate group attribution, and corrected submission requirements. All local documentation links resolve. |
| Testing and Reliability | 20% | 156 passing tests covering analytical results, invalid inputs, rank deficiency, numerical stability, gradient updates, and convergence. All notebooks execute; repeated fits reproduce results; package builds and installed-wheel checks pass. Published-head CI is green. No coverage percentage was measured. |
| Examples and Usability | 10% | Two verified README examples, four independent notebooks, ten rendered figures, bundled datasets, and locked reproduction instructions. |
| Repository Quality | 10% | Organized `src/`, `tests/`, `notebooks/`, `docs/`, workflow, lockfile, ignore rules, and 25 preserved commits through the reviewed head. The old contribution-pack note is archived in `docs/`, with its original text preserved. No duplicate notebooks or tracked cache/environment/build artifacts remain. |

Estimated compliance is strong across all five supplied categories, covering
100% of the rubric's stated weight. These are evidence assessments, not
awarded points. No verified missing technical deliverable was found.

## Current validation

- `uv sync --locked --dev` succeeds in the existing project environment:
  macOS, Python **3.11.4**, NumPy **2.4.6**, pytest **9.1.1**. The lockfile and
  dependency declarations remain unchanged; NumPy is the only runtime dependency.
- `uv run --locked pytest -v`: **156 passed**, zero failures or skips.
- All four notebooks execute from beginning to end in separate kernels using
  the project's `.venv` interpreter. Temporary audit instrumentation blocks
  external socket connections, checks the installed package path, records
  scaler/model fits, and verifies their actual training inputs.
- Every scaler learns only its training rows. Notebook 04 uses a fresh scaler
  and MLForge estimator for each fold, validates every training row exactly
  once, and excludes outer test rows from every fold. Baseline predictions use
  only fold training labels. Independent calculations reproduce all fold scores.
- Repeated Linear and Logistic Regression fits reproduce coefficients,
  intercepts, predictions, and Logistic Regression loss histories within
  numerical tolerances. All ten PNG figures decode and render correctly.
- Useful saved notebook results and figures are preserved. Re-execution matches
  saved numerical/text results; the only extra stream was a one-time Matplotlib
  font-cache notice, which was not saved to the repository.
- Both Python examples copied from the README pass. The source distribution
  and wheel build successfully; installed-wheel imports and tests pass.
- GitHub Actions [run 43503882226](https://github.com/xinnnma/Cmor-Proj/actions/runs/43503882226)
  succeeds for the exact published head. Local unpublished changes have not
  triggered a new run. CI covers imports and tests; notebook execution and
  package builds were verified locally.

## Verified notebook results

| Notebook | Data and split | Reproduced result | Figures |
| --- | --- | --- | --- |
| 01 — Data exploration | Diabetes: 442 rows, 10 features; 353 train / 89 test | No missing training values; training-only statistics, correlations, and standardization. | 3 |
| 02 — Linear Regression | Raw Diabetes, same split | R² **0.4526**, MSE **2900.1936**, RMSE **53.8534**, MAE **42.7941**; training-mean baseline MAE **64.0065**, RMSE **73.2225**. | 3 |
| 03 — Logistic Regression | Breast Cancer: 569 rows, 30 features; 455 train / 114 test, stratified | Accuracy **96.49%** (110/114), macro F1 **0.9627**, malignant recall **0.9762**; confusion matrix **`[[41, 1], [3, 69]]`**. | 3 |
| 04 — Cross-validation | Five folds within each outer training partition | Diabetes R² **0.4804 ± 0.0457**; Breast Cancer accuracy **0.9780 ± 0.0110**. Final holdout results match notebooks 02–03. | 1 |

Cross-validation ± values are sample fold standard deviations (`ddof=1`),
not confidence intervals. Diabetes loads with `scaled=False`; all splits use
`random_state=42`. Classification labels and confusion-matrix order are
**0 = malignant, 1 = benign**, with true labels as rows. One malignant case
is predicted benign. Training cross-entropy decreases from **0.693147** to
**0.050878** over 2,000 updates at learning rate 0.1.

## Limitations and submission readiness

The models retain their documented linear structure. Logistic Regression is
binary, unregularized, and uses fixed updates and a fixed 0.5 threshold;
preprocessing remains the caller's responsibility. Cross-validation measures
variation within these training partitions, not performance on new populations.
There is no hyperparameter search or clinical validation.

Local validation used macOS/Python 3.11.4; published CI verifies Ubuntu/Python
3.12. Other declared Python versions and platforms were not tested. Initial
dependency installation needs package-index access or a populated cache;
notebook execution needs no external datasets or downloads.

The reviewed project is technically ready for submission review. The local
corrections must still be approved, committed, and published before GitHub
contains this integrated documentation. No presentation delivery remains due.
