# snake_ai_legacy

Curated import of the legacy INDEGO Snake AI project from:

- `projects/legacy/indego_snake_game/Snake-ai-main`

This package preserves original file names and behavior for reproducibility while
exposing a stable module entrypoint:

```bash
PYTHONPATH=game_forge/src python -m snake_ai_legacy --list
PYTHONPATH=game_forge/src python -m snake_ai_legacy --variant classic
PYTHONPATH=game_forge/src python -m snake_ai_legacy --variant standalone
PYTHONPATH=game_forge/src python -m snake_ai_legacy --variant supersnake
```

Notes:

- `classic` requires `pygame`.
- `standalone` imports additional scientific/ML dependencies.
- `supersnake` requires `pygame_gui`.
- Source provenance remains immutable under `projects/legacy`.

