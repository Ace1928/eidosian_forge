# Developer Dependencies

> _"Development tools shape the quality and clarity of the codebase."_

The following packages are required only for development and testing. They are
not needed for running Word Forge in production environments.

The main `requirements.txt` lists runtime packages such as `networkx` and
`numpy` that are needed for graph and vector features.

```text
black
isort
mypy
pytest
pytest-cov
```

Each tool serves a distinct purpose:

- **black** – Enforces a consistent code style across the project.
- **isort** – Automatically sorts imports to reduce merge conflicts.
- **mypy** – Provides optional static type checking.
- **pytest** and **pytest-cov** – Run tests and measure code coverage.

Install them with `pip` using the `--requirement` option if desired:

```bash
pip install -r requirements.txt  # runtime dependencies
pip install black isort mypy pytest pytest-cov  # development only
```
