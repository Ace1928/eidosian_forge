# 🎮 Game Forge Installation

Game development utilities, state machines, and interactive systems.

## Quick Install

```bash
pip install -e ./game_forge

# Verify
python -c "from game_forge import GameEngine; print('✓ Ready')"
```

## CLI Usage

```bash
# Launch game engine
game-forge run <game_file>

# Debug mode
game-forge debug <game_file>

# Help
game-forge --help
```

## Python API

```python
from game_forge import GameEngine, GameState

# Create game engine
engine = GameEngine()
engine.run()
```

## Dependencies

- `pygame` (optional) - Graphics rendering
- `eidosian_core` - Universal decorators and logging

