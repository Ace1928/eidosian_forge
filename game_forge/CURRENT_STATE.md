# Current State: game_forge

**Date**: 2026-01-20
**Status**: Experimental

## 📊 Metrics
- **Dependencies**: `pygame`, `numpy`.
- **Projects**: Multiple sub-projects in `src/` (ECosmos, Stratum, gene_particles, etc.).

## 🏗️ Architecture
Collection of independent simulation modules.

## 🐛 Known Issues
- Some sub-projects still lack `__main__.py` entrypoints (notably `chess_game` and `algorithms_lab`).
- `eidosian_core` lives in `lib/eidosian_core`; ensure `lib` is on `PYTHONPATH` for direct runs.
