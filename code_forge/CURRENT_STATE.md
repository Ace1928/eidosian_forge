# Current State: code_forge

**Date**: 2026-01-20
**Status**: Transitioning

## 📊 Metrics
- **Legacy Content**: Contains `forgeengine` (Narrative Engine).
- **New Direction**: Pivoting to Static Analysis & Refactoring.
- **Dependencies**: Added `libcst`, `rope`, `tree-sitter`.

## 🏗️ Architecture
Currently a mix of:
1.  `forgeengine/`: A narrative/chat engine (Legacy).
2.  `code_core.py`: Likely a placeholder or simple utility.

## 🐛 Known Issues
- `forgeengine` seems misplaced; strictly speaking, it belongs in `llm_forge` or `game_forge`.
- No actual Code Forge implementation exists yet (CLI is defined but not implemented).
- `install.sh` is legacy.