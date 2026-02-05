# Current State: game_forge

**Date**: 2026-02-05
**Status**: Experimental

## 📊 Metrics
- **Dependencies**: `pygame`, `numpy`.
- **Projects**: Multiple sub-projects in `src/` (ECosmos, Stratum, gene_particles, etc.).

## 🏗️ Architecture
Collection of independent simulation modules.

## 🐛 Known Issues
- Optional GPU backends (CuPy/OpenCL) are not always available in headless environments; related tests skip when deps are missing.
- `eidosian_core` lives in `lib/eidosian_core`; ensure `lib` is on `PYTHONPATH` for direct runs.
