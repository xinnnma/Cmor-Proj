# Notebooks

These notebooks use the installed MLForge package and scikit-learn's bundled
datasets. The model implementations are imported from `mlforge`; they are not
copied into notebooks or replaced by scikit-learn estimators.

| Notebook | Purpose |
| --- | --- |
| [01_data_exploration.ipynb](01_data_exploration.ipynb) | Explore raw Diabetes features, inspect statistics and missing values, visualize relationships, and demonstrate training-only scaling. |
| [02_linear_regression.ipynb](02_linear_regression.ipynb) | Train MLForge Linear Regression on Diabetes; evaluate R², MSE, RMSE and MAE; inspect prediction and residual plots; compare with a training-mean baseline. |
| [03_logistic_regression.ipynb](03_logistic_regression.ipynb) | Train MLForge Logistic Regression on Breast Cancer; inspect probabilities, accuracy, class-level precision, recall, F1, the confusion matrix and training loss. |
| [04_cross_validation.ipynb](04_cross_validation.ipynb) | Evaluate both models with five-fold cross-validation on the training partition, fit preprocessing separately within each fold, and compare against simple baselines. |

## Install the environment

From the MLForge repository root, run:

```bash
uv sync --locked --dev
```

The development group includes pandas, matplotlib, scikit-learn, nbconvert,
and the existing IPython kernel and pytest tools. NumPy remains the package's
only runtime dependency. No JupyterLab or browser server is needed for VS Code.
Installing dependencies initially needs package-index access; running the
notebooks afterward does not need internet or dataset downloads.

## Run in VS Code

1. Open the MLForge folder and use the Python and Jupyter extensions.
2. Open any of the four notebooks.
3. In **Select Kernel**, choose **Python Environments** and the interpreter
   inside MLForge's `.venv` folder.
4. Restart the kernel and choose **Run All**. Save the notebook to retain
   updated outputs and figures.

Each notebook runs independently from top to bottom. Read them in numbered
order for the full workflow. If importing `mlforge` fails, rerun the setup
command and confirm that the selected kernel belongs to this project's `.venv`.
Do not add source directories to `sys.path` or copy model code into a notebook.

## Optional terminal execution

To make the environment explicit for Jupyter's command-line tools, register
the project kernel on your machine and then execute all four notebooks:

```bash
uv run python -m ipykernel install --user --name mlforge --display-name "Python (MLForge)"
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=mlforge \
  --ExecutePreprocessor.timeout=180 notebooks/*.ipynb
```

Kernel registration is local to your machine. The execution command updates
the saved outputs in the four notebook files. The same notebooks can also
be run interactively in VS Code without registering a command-line kernel.

## Reproducibility and data handling

- Install the locked environment with `uv sync --locked --dev`.
- Each split uses 20% test data and `random_state=42`; classification also uses
  stratification to preserve class proportions.
- Diabetes uses raw features (`scaled=False`) rather than the loader's
  normalization over all observations.
- Exploration uses training rows. Scalers learn training means and standard
  deviations only, then transform the held-out features without refitting.
- MLForge models fit training data only. Test targets are used only for
  evaluation, and model settings are fixed before evaluation.
- Breast Cancer labels are **0 = malignant, 1 = benign**; probability columns
  follow that order. Both datasets are bundled with scikit-learn.
- Keep the saved metrics, concise tables, and figures for GitHub viewing.
  Exact plot appearance may vary by platform; numerical results should agree
  within ordinary floating-point tolerances in the locked environment.

Notebooks 01–03 use one fixed train/test split. Notebook 04 adds five-fold cross-validation
within the training partition, with a fresh scaler and model fitted in each fold.
It does not perform a hyperparameter search. These demonstrations do not establish
performance on new populations.

## Reading the saved results

The four notebooks contain ten figures in total: three for data exploration,
three for Linear Regression, three for Logistic Regression, and one for cross-validation.

| Notebook | Saved evaluation results |
| --- | --- |
| Linear Regression | R² **0.4526**, MSE **2900.1936**, RMSE **53.8534**, and MAE **42.7941** on 89 test rows. The training-mean baseline has MAE **64.0065** and RMSE **73.2225**. |
| Logistic Regression | Accuracy **96.49%** (110/114), macro F1 **0.9627**, and malignant-class recall **0.9762**. |
| Cross-validation | Five-fold training-partition comparisons report validation scores for both models and their simple baselines, including mean scores and fold standard deviations. |

R² and accuracy measure different tasks. R² compares regression squared error
with a test-mean baseline; accuracy counts correct binary labels. The Breast
Cancer confusion matrix is `[[41, 1], [3, 69]]`, with true labels as rows and
predicted labels as columns in the order malignant, benign. It includes one
malignant case classified as benign, so the overall accuracy does not describe
every kind of error. The training-loss plot shows optimization progress; it
does not measure performance on unseen data.
