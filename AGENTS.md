# Agent Instructions

## Project Status

MLForge — A Machine Learning Package Built from Scratch is a Python 3.11+ group course project using a `src/mlforge/` package layout, NumPy, uv, Hatchling, and pytest. Its local folder is `/Users/third/Desktop/MLForge`, and its public repository is `https://github.com/xinnnma/Cmor-Proj`. It includes two algorithms, 156 verified tests, and four executed notebooks using bundled Diabetes and Breast Cancer datasets, including five-fold cross-validation with preprocessing fitted within each fold. Preserve both algorithms, teammates' contributions and authorship, tests, and useful notebook outputs. Final integration is recorded in PR #2 from `final-integration` into `main`, approved for a normal merge commit. Subsequent work, commits, merges, and pushes require explicit instructions. The final deliverable is the GitHub repository; no presentation is required, per the team's November 13, 2026 instructor update. Audits, the integration report, and optional historical presentation notes belong in `docs/`.

## Working Rules

- Inspect relevant files and nearby tests before making changes.
- Keep changes focused on the requested task and preserve unrelated user changes.
- Prefer existing project patterns and dependencies over introducing new abstractions or tools.
- Update or add tests when behavior changes, and run the narrowest relevant checks after editing.
- If a check cannot be run, state why and identify the remaining validation needed.
- Do not commit changes or create branches unless explicitly requested.
- Do not add secrets, generated artifacts, or unrelated configuration to the repository.
- Work only in MLForge; do not modify the professor's course repository or the Benzhou website.
- Do not change the configured Git author identity or publish to GitHub without the user's instructions. Preserve existing commit history and historical authorship.
- Keep real tests in `tests/` and notebooks in `notebooks/`. Do not add placeholder tests that imply algorithms work.
- Notebook scalers and models must fit training rows only. Use installed MLForge models; scikit-learn is for bundled datasets and utilities. NumPy remains the only runtime dependency; notebook tools belong in the development group.

## Verified Commands

- `uv sync --locked --dev` installs the package and development dependencies into `.venv`.
- `uv build` builds the source distribution and wheel.
- `uv run pytest -v` runs both models' tests against analytical solutions, deterministic datasets, and invalid inputs. Logistic Regression also checks numerical stability and convergence.
- Use the project `.venv` kernel to execute notebooks from top to bottom. See `notebooks/README.md` for VS Code and command-line setup. Preserve useful outputs and figures, and remove saved errors and unnecessary execution metadata.

## Keeping These Instructions Current

When the project adopts a language, framework, or standard validation commands, update this file with verified project-specific guidance. Keep instructions concise and avoid duplicating details already documented in the project.
