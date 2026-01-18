# TODO: Script Library Standardization and Upgrades

Status: completed.

## Completion notes
- All listed items have been addressed with code changes, docs, and safety fixes.
- Tests now run via `python3 -m unittest discover -s tests`.
- Pytest failed due to a missing third-party plugin dependency (`zstandard` via `langsmith`).
- Profiling performed for help-paths of `smart_publish` and `forge_builder` using `cProfile`.

If you want additional profiling runs, provide specific commands and inputs to target.
