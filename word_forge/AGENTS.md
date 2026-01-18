# Repository Guidelines

## Project Structure & Module Organization
- `src/word_forge/` is the core package, organized by domain (`queue/`, `database/`, `parser/`, `vectorizer/`, `emotion/`, `graph/`, `conversation/`, `workers/`).
- `tests/` contains unit and integration tests that exercise ingestion, graph/vector flows, and conversation pipelines.
- `data/` holds runtime artifacts like `data/exports/` and `data/visualizations/` for graph outputs.
- `docs/` includes migration and architecture notes; `logs/` and `backups/` are used for local runs.
- SQLite DBs such as `word_forge.sqlite` may live at repo root or under custom paths for local experiments.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and activate a local virtualenv.
- `pip install -r requirements.txt`: install runtime/dev dependencies.
- `python -m pytest`: run the full test suite.
- `word_forge start hydrogen --minutes 0.1 --workers 2`: run a short crawl/ingest loop.
- `word_forge graph build --timeout 30`: build the graph from DB data.
- `word_forge vector index --embedder sentence-transformers/all-MiniLM-L6-v2`: build vector indices.

## Coding Style & Naming Conventions
- Python 3.8+ with type hints; follow the stricter `mypy` settings in `pyproject.toml`.
- Format with `black` and sort imports with `isort` (`profile = "black"`).
- Use descriptive class names (e.g., `GraphManager`, `QueueWorker`) and snake_case for functions.

## Polishing & Perfection Prompts
- Apply the Eidosian refinement guides in `code_polishing_prompt.md` and `code_perfection_prompt.md` when tightening code quality.
- Keep changes minimal while enforcing typed interfaces, explicit error handling, and the Napoleon/Google docstring style requested in those prompts.
- When in doubt, favor the stricter guidance from the prompts and document non-obvious decisions inline with existing style.

## Testing Guidelines
- Use `pytest`; name files `test_*.py` and keep tests close to the feature they exercise.
- Integration tests should validate real ingestion, graph, and vector flows without mocks or stubs.
- When adding new features, include failure-mode coverage (missing data, queue state, DB errors).

## Commit & Pull Request Guidelines
- Commit messages are imperative and scoped (examples: `fix: start queue lifecycle`, `docs: update CLI flags`).
- PRs should include a short summary, test results, and any data/DB migration notes.
- If a change alters runtime artifacts (graphs, vectors), document the regeneration command.

## Local Data & Configuration Tips
- Prefer `--db-path` for isolated test runs; avoid overwriting shared DBs.
- Keep generated outputs in `data/` or `logs/` rather than committing them.
