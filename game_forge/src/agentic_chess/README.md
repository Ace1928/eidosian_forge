# Agentic Chess

Agent-versus-agent chess runner with optional Agent Forge integration.

## Usage

```bash
source /home/lloyd/eidosian_forge/eidosian_venv/bin/activate
PYTHONPATH=game_forge/src:agent_forge/src python -m agentic_chess --list-agents
PYTHONPATH=game_forge/src:agent_forge/src python -m agentic_chess --white random --black agent-forge
python game_forge/tools/agentic_chess_benchmark.py --games 5 --max-moves 60 --output game_forge/tools/artifacts/agentic_chess_benchmark.json
```

## Options

- `--white` / `--black`: select `random` or `agent-forge` players.
- `--agent-forge-policy`: move policy for the Agent Forge player (`capture` or `random`).
- `--agent-forge-memory`: write agent_forge task memory to a directory.
- `--agent-forge-no-git`: disable git-backed memory if you set `--agent-forge-memory`.
- `--pgn-output`: write the match PGN to a file.

## State Management

- Each match starts from a fresh `python-chess` board state.
- The match RNG is seeded once per game (if provided) and shared across agents.
- Illegal moves terminate the game immediately.
- Moves are recorded as SAN and PGN is available when `--pgn-output` is set.

## Dependencies

- `python-chess`

Install:

```bash
source /home/lloyd/eidosian_forge/eidosian_venv/bin/activate
pip install python-chess
```

## Notes

- `eidosian_monorepo/` is legacy scaffolding; the active runtime lives in this package.
