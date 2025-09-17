# Contributing

Thank you for your interest in improving this project! This guide explains how to set up your environment, propose changes, and open pull requests.

## Development Environment
- Python: 3.12 (file: `.python-version`)
- Dependency manager: uv

Setup
- Install uv: https://docs.astral.sh/uv/
- Install project deps: `uv sync`
- Optional dev tools: `uv sync --dev` (if you have dev-dependencies configured) or `uv add --dev black flake8 pytest`

Convenience commands
- Train: `uv run python main.py train --data data/balanced_ai_human_prompts.csv`
- Evaluate: `uv run python main.py evaluate --data data/balanced_ai_human_prompts.csv`
- Tune: `uv run python main.py tune --data data/balanced_ai_human_prompts.csv`
- GUI: `uv run python main.py gui`

## Branching and Workflow
- Create a feature branch off `main`:
  - `git checkout -b feat/<short-name>` for new features
  - `git checkout -b fix/<short-name>` for bug fixes
  - `git checkout -b docs/<short-name>` for docs-only changes
- Keep changes focused and small; prefer multiple small PRs over a very large one.

## Commit Messages
- Use clear, meaningful messages.
- Recommended style (light Conventional Commits):
  - `feat: add GUI prediction window`
  - `fix: handle empty input in dataset loader`
  - `docs: expand README usage section`

## Code Style
- Python formatting: `black .`
- Linting: `flake8 .`
- Keep functions short, with clear names and docstrings where helpful.

## Tests
- Add tests (if applicable) alongside new functionality.
- Run tests locally: `uv run pytest`

## Data and Models
- Keep the dataset under `data/`.
- Do not commit large binary artifacts where avoidable. Trained models are ignored by default via `.gitignore` (see `models/*.joblib`). If you need to share a model, use releases or storage links.

## Pull Requests
- Ensure the branch is up to date with `main`.
- Include a concise description of changes and any screenshots for UI changes.
- Reference related issues if applicable.

Thanks for contributing!
