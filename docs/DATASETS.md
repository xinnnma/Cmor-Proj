# Dataset and prediction contracts

These datasets are educational examples bundled with the locked scikit-learn
installation. No external data download is required after installing dependencies.
We import datasets and utilities from scikit-learn; both trained model classes
come from MLForge. This project makes no clinical or causal claim.

Descriptions below follow the loaders' bundled `DESCR` metadata. The loader links
provide source references; metadata that is not established there is left unknown.
We do not redistribute a new standalone dataset or assert a new dataset license.
Retain source attribution and check upstream terms before separately redistributing data.

## Diabetes regression

| Field | Contract |
| --- | --- |
| Loader | `load_diabetes(as_frame=True, scaled=False)` |
| Size | 442 patient rows, ten baseline features |
| Observational unit | One patient row in this teaching dataset |
| Target | Supplied numerical disease-progression measure one year after baseline |
| Prediction time | Baseline measurements are available; the later progression response is not |
| Task and loss | Continuous prediction; least-squares fitting |
| Reported metrics | R², RMSE, MAE, MSE; RMSE and MAE retain supplied target units |
| Baseline | Mean of fitting-partition target values; optimal constant for squared loss, not necessarily absolute loss |
| Outer split | 353 development / 89 test rows; 80/20 with random state 42 |
| CV | Five shuffled folds inside development; validation sizes 71, 71, 71, 70, 70 |
| Missingness | Checked in notebooks; complete bundled features, with no imputation fitted |
| Interpretation limit | The supplied target unit is not assigned an invented physical or clinical meaning |

Feature dictionary:

| Feature | Meaning from bundled metadata | Unit/encoding caution |
| --- | --- | --- |
| age | Age | Years |
| sex | Coded sex variable | Discrete coding; do not interpret as continuous physiology |
| bmi | Body mass index | Retain the supplied measurement scale |
| bp | Average blood pressure | Exact unit is not stated in the bundled description used here |
| s1 | Total serum cholesterol | Preserve supplied units |
| s2 | Low-density lipoproteins | Preserve supplied units |
| s3 | High-density lipoproteins | Preserve supplied units |
| s4 | Total cholesterol / HDL ratio | Ratio |
| s5 | Described as possibly log serum triglycerides | Transformation meaning is qualified in source metadata; do not overinterpret |
| s6 | Blood sugar level | Preserve supplied units |

`scaled=False` avoids the loader's default scaling over the complete table.
StandardScaler learns feature means and population-standard-deviation scales
(`ddof=0`) from the fitting partition only. The response is not standardized.
Coefficient interpretation in supplemental notebook 05 is per training SD, conditional on the
other features, and does not establish causal effects.

Sources: [loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html);
[original source referenced by the loader](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html).

## Wisconsin Diagnostic Breast Cancer classification

| Field | Contract |
| --- | --- |
| Loader | `load_breast_cancer(as_frame=True)` |
| Size | 569 observations, 30 numeric features; 212 malignant and 357 benign labels |
| Observational unit | One diagnostic breast-mass sample represented by image-derived measurements |
| Features | Cell-nucleus characteristics from digitized fine-needle-aspirate images |
| Target | Recorded diagnostic category, not a future disease-progression horizon |
| Package encoding | **0 = malignant; 1 = benign** |
| Probability columns | Column 0 is P(malignant); column 1 is P(benign) |
| Package decision rule | Predict class 1 when P(benign) >= 0.5, including equality |
| Task and loss | Binary probability model; mean binary cross-entropy |
| Outer split | Stratified 455 development / 114 test rows; random state 42 |
| New probability experiment | Stratified 341 inner-fit / 114 validation rows drawn only from the 455 development rows |
| CV | Five stratified development folds: 364 fitting / 91 validation rows each |
| Baselines | Training majority label for accuracy; training event prevalence for probability losses |
| Missingness | Checked in notebooks; no missing training values in the bundled table |

Ten measurement families describe radius, texture, perimeter, area, smoothness,
compactness, concavity, concave points, symmetry, and fractal dimension. Their
mean, standard error, and worst-value summaries form the 30 input columns.
The exact physical units/acquisition details are not assumed from the column
names. Preserve the loader's column order and fitted transformation.

For the supplemental probability analysis only, we define `event = (y == 0)` and use
probability column 0, so an event means malignancy. The illustrative threshold
rule flags malignancy at `p_event >= t`; this states its own equality convention
and does not replace the package's `predict` behavior. Thresholds are fixed
examples, not optimized or applied to select a test-set winner.

Sources: [loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html);
[UCI dataset record](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
Creators credited in bundled metadata: William H. Wolberg, W. Nick Street,
and Olvi L. Mangasarian.

## Shared information boundary and limitations

- Fit every scaler only on its current fitting rows, including within CV.
- Retain feature order and meaning. MLForge validates feature count and numeric
  values, but does not retain DataFrame column names to detect reordered columns.
- Development means the original outer training partition; validation means
  rows temporarily withheld inside that partition. These roles are distinct
  from the original test partition.
- The original holdout scores were viewed before this revision. Reproducing them
  is not a new independent evaluation. No new configuration is selected here.
- Random-row splits approximate exchangeable observations within each teaching
  dataset. Site, time, repeated-entity linkage, and external population metadata
  are insufficient here to verify stronger transfer assumptions.
- A random seed supports repeatability; it does not validate sampling assumptions.
- Do not infer causal effects, clinical usefulness, or calibrated deployment risk.
- Report model, split, loader, and software versions with results. The lockfile
  selects versions according to Python/platform; different supported runtimes
  need not install identical binary dependency versions.

## Course connections

Textbook Chapter 11 motivates the prediction contract and baseline; Chapter 12
motivates regression and conditioning; Chapter 14 separates probability, ranking,
and decisions; Chapter 24 distinguishes development selection from final evaluation.
