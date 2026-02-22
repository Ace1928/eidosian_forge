# TODO: file_forge

## 🚨 High Priority
- [x] **Refactor**: Move `file_core.py` to `src/file_forge/core.py`.
- [x] **Cleanup**: Cleaned up structure.

## 🟡 Medium Priority
- [x] **Feature**: Add fuzzy file finding (glob via `find_files`).

## 🟢 Low Priority
- [x] ~~**Optimization**: Use `ripgrep` (via subprocess) for faster searching.~~ (`search_content` now uses `rg` when available with Python fallback)
