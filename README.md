# Human vs AI Essay Classifier

Binary classifier to distinguish AI-generated vs human-written essays using the Kaggle dataset `balanced_ai_human_prompts.csv`. Built with scikit-learn and managed with uv for reproducible environments.

## Team Members

| AC.NO | Name | Role | Contributions |
|----|------|------|---------------|
| 1 | Your Name | Lead | Data prep, modeling |
| 2 | Teammate | Analyst | EDA, visualization |
| 3 | Teammate | Engineer | Evaluation, packaging |

## Installation and Setup

Prerequisites
- Python 3.12+
- uv (https://docs.astral.sh/uv/)

Steps
- Clone or open this folder.
- Install dependencies with uv:
  - `uv sync`
- Optional: activate the venv for your shell
  - `uv venv --python 3.12`
  - `uv run python --version`

## Project Structure

```
project/
- README.md
- pyproject.toml
- .python-version
- main.py               # CLI entry point
- src/
  - data/
    - dataset.py        # Load/split utilities
  - models/
    - model.py          # Pipeline, train/eval
  - utils/
    - io.py             # Save/load model
    - metrics.py        # Metrics helpers
- notebooks/
- data/
  - balanced_ai_human_prompts.csv  # Dataset
- docs/
```

## Usage

Train (uses the CSV in `data/` by default):
- `uv run python main.py train --data data/balanced_ai_human_prompts.csv --model-path models/model.joblib`

Evaluate on the test split:
- `uv run python main.py evaluate --data data/balanced_ai_human_prompts.csv --model-path models/model.joblib`

Predict a single text input:
- `uv run python main.py predict --model-path models/model.joblib --text "Your essay text here..."`

Hyperparameter tuning (GridSearchCV over TF-IDF + LinearSVC):
- `uv run python main.py tune --data data/balanced_ai_human_prompts.csv --cv 3 --jobs -1 --model-path models/model_tuned.joblib`

Run the EDA notebook:
- `uv run jupyter notebook notebooks/01_eda.ipynb`

Launch GUI to paste text and get predictions:
- `uv run python main.py gui`

Notes
- The script creates a stratified train/test split (80/20) on-the-fly from the provided CSV. The label column is `generated` ("1" = AI-generated, "0" = human).
- The model is a scikit-learn pipeline: TF-IDF (word + bigram) + Linear SVM.

## Results

After training, the script prints accuracy, precision, recall, and F1 on the test set and saves the model to `models/model.joblib`.

## Contributing

- Create a feature branch: `git checkout -b feature-name`
- Commit with clear messages and keep changes focused.
- Open a pull request for review.
