# Falling Sand

Production-quality tooling scaffold with indexing, testing, and profiling.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .[dev]
```

## Verification
Run the full verification suite after any change:

```bash
scripts/verify.sh
```

This runs:
- Unit tests
- Integration tests
- Code indexing (JSON output)
- Profiling (cProfile output)
- Benchmark suite (JSON output, indexer + simulation)
- SQLite ingestion (database output)
- Trend reporting (JSON output)
- Optional linting/type checks (when installed)

## Indexing
Generate a structured index of source and test symbols:

```bash
python -m falling_sand.indexer --output artifacts/index.json
```

The JSON output is designed to be ingested by a future database for code, test, and profiling metadata.

Pass `--exclude-dir` to ignore additional directories or `--allow-missing-tests` if tests are optional.

## Engine quick start

```bash
python - <<'PY'
from falling_sand.engine import Material, VoxelConfig, World, step_world
from falling_sand.engine.simulation import SimulationConfig

config = VoxelConfig()
world = World(config=config)
world.set_voxel((5, 5, 5), Material.GRANULAR)
sim_config = SimulationConfig()
print(step_world(world.chunks, config=sim_config))
PY
```

To enable rendering, install the optional extra:

```bash
.venv/bin/python -m pip install .[render]
```

Run the interactive demo:

```bash
python scripts/demo_panda3d.py
```

Controls: `wasd` move, `mouse` look, `mouse1` place, `mouse3` erase, `scroll` zoom, `1-4` material.
The demo launches fullscreen with mouse capture enabled by default.

CLI entrypoints:

```bash
falling-sand demo
falling-sand bench
falling-sand index
falling-sand ingest
falling-sand report
```

Module entrypoint:

```bash
python -m falling_sand demo
```

Instanced rendering example:

```bash
python - <<'PY'
from falling_sand.engine import Material, VoxelConfig, World, InstancedVoxelRenderer

config = VoxelConfig()
world = World(config=config)
world.set_voxel((1, 1, 1), Material.SOLID)

renderer = InstancedVoxelRenderer(config)
renderer.render_chunks(world.iter_chunks())
renderer.run()
PY
```

Chunk streaming and ray-based spawning are built into the demo; see `docs/ENGINE.md` for details.
The demo includes an overlay with FPS, chunk count, and material info.
Terrain generation uses fractal noise to build a large 10k x 10k world on demand.
Rendering uses dirty-chunk rebuilds with distance/frustum culling for higher performance.

## Database ingestion

```bash
python -m falling_sand.ingest --index artifacts/index.json --db artifacts/index.db
```

## Reporting

```bash
python -m falling_sand.reporting --db artifacts/index.db --output artifacts/report.json
```

## Documentation
- `docs/DATABASE.md`
- `docs/INDEXING.md`
- `docs/ENGINE.md`
- `docs/REPORTING.md`
- `docs/PIPELINE.md`
- `docs/SCHEMA.md`
- `docs/TESTING.md`
