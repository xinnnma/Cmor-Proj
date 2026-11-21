# Findings, limitations and verification of the supplement

This document synthesizes the original two-model project and the additions-only
analysis. It does not modify or repair the original implementations. Read it with
[PROJECT_GUIDE.md](../PROJECT_GUIDE.md) and [DATASETS.md](DATASETS.md).

## Research questions and findings

**Can simple transparent models improve on constant predictions?** The original
Diabetes holdout uses 353 fitting and 89 test rows. Linear Regression reports
R² 0.4526, RMSE 53.8534 and MAE 42.7941. The training-mean baseline has RMSE
73.2225 and MAE 64.0065. This supports an improvement on the displayed split,
not accurate individual predictions in general. Five of the 89 absolute residuals
exceed 100 supplied target units. These summaries refer to the original holdout;
notebook 05 adds training coefficient/conditioning analysis, not a new test set.

The original Breast Cancer experiment uses 455 fitting and 114 test rows. It
classifies 110/114 correctly (96.49%), with confusion matrix `[[41,1],[3,69]]`
ordered malignant (0), benign (1), true labels in rows. One malignant observation
is missed. Macro F1 is 0.9627 and malignant recall is 41/42 = 0.9762. Accuracy
alone hides this asymmetric practical error. The example is not a clinical model.

**What do the fitted coefficients and optimization imply?** Notebook 05 shows
that the training diabetes design has full column rank (11/11) and its measured
condition number decreases from approximately 7065 to 21 after standardization.
The BMI coefficient is about 25.6071 target units per training SD, conditional on
other encoded features. Correlated predictors and observational data prevent a
causal interpretation. The intercept 153.7365 is the fitted response at training
feature means. Training R² 0.5279 and RMSE 53.5588 are in-sample summaries.

The logistic demonstration recovers the gradient used by the unchanged estimator
from its first and second public `fit` trajectories and compares it with central
finite differences. This tests actual training behavior at a nonzero parameter
vector without requiring a private gradient API. It does not prove convergence
for every step size or dataset. Fixed iterations, unregularized weights and
possible separation remain important limitations.

**Are ranking, probability quality and decisions the same?** No. Notebook 06
fits a fresh classifier on 341 rows drawn from the original 455 development rows
and evaluates on the remaining 114 development-validation rows. These are not
the original 114 test observations. Fixed settings remain learning rate 0.1 and
2,000 iterations, with preprocessing fitted only on inner-fitting rows.

| Development-validation metric | MLForge | Training-prevalence baseline |
| --- | ---: | ---: |
| Log loss (lower is better) | 0.0571 | 0.6627 |
| Brier score (lower is better) | 0.0157 | 0.2349 |
| ROC AUC (higher is better) | 0.9984 | 0.5000 |

The event is explicitly malignancy: `y == 0`, using probability column 0.
High AUC describes ranking, not accurate risk probabilities. The five fixed
calibration bins contain 68, 2, 3, 1 and 40 observations. Intermediate bins are
far too sparse for strong calibration conclusions; Wilson intervals illustrate
uncertainty in each observed event fraction under binomial sampling assumptions.
They are not simultaneous bands or external validation.

Thresholds 0.20, 0.35, 0.50, 0.65 and 0.80 are examples, not an optimization grid
used to choose a winner. The malignant-event rule uses `p_event >= threshold`;
its equality convention differs from the package's benign-class `>= 0.5` rule.
No operating threshold is selected, and none is evaluated as a new test winner.

**How variable is performance across development folds?** Notebook 06 uses fresh
models/scalers within the original five-fold scheme and checks disjointness,
coverage and training-only scaling. Results below summarize the five validation
folds; SD is the sample SD (`ddof=1`).

| Metric | Fold mean | Fold SD |
| --- | ---: | ---: |
| Diabetes R² | 0.4804 | 0.0457 |
| Diabetes RMSE | 55.3946 | 2.6402 |
| Diabetes MAE | 45.0482 | 1.9818 |
| Breast Cancer accuracy | 0.9780 | 0.0110 |
| Breast Cancer malignant recall | 0.9647 | 0.0322 |
| Breast Cancer log loss | 0.0685 | 0.0333 |

Average training-minus-validation gaps are approximately 0.0505 for R² and 0.0110
for accuracy. These are descriptive differences, not proof of a specific cause.
The mean validation baseline R² is -0.0273; majority-label accuracy is 0.6264.
The new single validation experiment and CV averages use different fits and
subsets, so their metrics need not be identical.

## Evaluation qualifications

The original holdout results had already been viewed before these additions.
Any original wording suggesting a newly untouched, one-time final test should
be read with that qualification. Notebook 06 excludes the original test rows
from all its fitting and new performance evaluation; that does not undo previous
exposure. No hyperparameter or threshold is selected in these additions.

CV folds share fitting observations; their scores are dependent. Fold SD is not
a confidence interval or an independent-replicate standard error. Random-row
splits assume sufficiently exchangeable observations; these experiments do not
validate performance at a different hospital, time period or population. No
causal effects, deployment calibration, or clinical utility are established.

## Known original numerical limitation — not repaired

The original Linear Regression checks finite inputs, but finite values can still
overflow during prediction or when squaring residuals. For example, the following
original public API calls expose the issue on the tested runtime:

```python
import numpy as np
from mlforge import LinearRegression

model = LinearRegression().fit([0., 1., 2.], [1., 3., 5.])
with np.errstate(over="ignore", invalid="ignore"):
    print(model.predict([1e308]))  # nonfinite prediction
    print(model.score([0., 1., 2.], [-1e200, 0., 1e200]))  # NaN
```

Suppressing the warning above makes the demonstration readable; it is not a fix.
This does not occur in the supplied ordinary-scale dataset demonstrations. Those
successful examples nevertheless do not establish safety for arbitrary finite
inputs. A future source edit should check solver/prediction finiteness, preserve
previous fitted state on numerical solver failure and scale R² arithmetic before
squaring. Such fixes were deliberately excluded to preserve every original file.

Both models also depend on callers preserving feature order and preprocessing.
They do not store DataFrame column names to detect reordering. No automatic
regularization, early stopping or multiclass extension is introduced here.

## Verification record

Preparation date: November 20, 2026. Validation is against the exact original
uploaded ZIP plus these seven new repository files, not the live main branch.

- Original files checked byte-for-byte against the uploaded ZIP after validation.
- All 166 pytest cases passed: 156 original and ten new mathematical cases.
- All six notebooks executed in separate Python/IPython processes with the
  original source installed; saved new notebook outputs contain no error outputs.
- The new notebook 05 contains numerical tables; notebook 06 contains two figures
  (ROC/calibration panels and fold-score panels). Both new figures inspected.
- Native Jupyter-kernel validation remains unverified in the preparation
  environment, which previously blocked kernel sockets. Saved supplemental
  notebook metadata explicitly records the process fallback.
- Existing workflow is unchanged. No remote CI, commit, push, merge or public
  repository visibility check is claimed by this supplement.

Before submission, follow the notebook execution instructions in
[the notebook guide](../notebooks/README.md) and review the resulting
tables and figures. Additional tests check residual orthogonality (two cases), 
full-rank affine scaling invariance (one), gradients recovered from public updates
(six) and an independent two-step recurrence (one). They do not import or require 
the earlier revision's private helper and do not claim to fix the numerical defect.

The original README's 156-test/four-notebook counts describe the original project.
With this supplement the total is 166 cases and six notebooks. Historical records
and authorship are unchanged. These improvements strengthen rubric evidence;
only the instructor can determine the grade and sufficient methodological breadth.
