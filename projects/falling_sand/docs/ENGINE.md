# Engine

Initial 3D falling-sand engine with a chunked voxel world and NumPy-based simulation.

## Concepts
- Voxel size: 0.1m (10cm)
- Chunk size: 10 voxels per axis (1m cube)
- Materials: air, solid, granular, liquid, gas
- Terrain: 10k x 10k blocks, 20 layers deep (procedural heightmap + water fill)

## Physics rules
- Granular: falls down, attempts diagonal descent when blocked.
- Liquid: falls down, spreads laterally when blocked.
- Gas: rises up, spreads laterally when blocked.
- Solid: immobile.

## Terrain generation
Procedural terrain uses fractal value noise with a configurable water level and soil depth.
Chunks are generated on-demand by the `TerrainGenerator` provider, keeping memory usage bounded
even for the 10k x 10k world.

## Chunk streaming
`ChunkStreamer` keeps a cubic radius of chunks active around a focus point. The demo binds the
streaming focus to the player so the world follows your movement inside the large terrain bounds.

## Usage

```bash
python - <<'PY'
from falling_sand.engine import Material, VoxelConfig, World, step_world
from falling_sand.engine.simulation import SimulationConfig

config = VoxelConfig()
world = World(config=config)
world.set_voxel((5, 5, 5), Material.GRANULAR)

sim_config = SimulationConfig()
for _ in range(3):
    stats = step_world(world.chunks, config=sim_config)
    print(stats)
PY
```

## Rendering
The optional Panda3D renderer visualizes voxels as instanced cubes with per-material colors and
transparency. Rendering now supports distance + frustum culling and only rebuilds dirty chunks to
reduce overhead on large worlds.

```bash
pip install .[render]
python - <<'PY'
from falling_sand.engine import Material, VoxelConfig, World
from falling_sand.engine.renderer_panda3d import Panda3DRenderer

config = VoxelConfig()
world = World(config=config)
world.set_voxel((1, 1, 1), Material.GRANULAR)

renderer = Panda3DRenderer(config)
renderer.render_chunks(world.iter_chunks())
renderer.run()
PY
```

## GPU instancing

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

## Interactive demo

```bash
python scripts/demo_panda3d.py
```

Controls:
- `1` granular, `2` liquid, `3` gas, `4` solid
- `mouse1` spawn particles, `mouse3` erase particles
- `mouse` look (captured in fullscreen)
- `scroll` zooms the camera
- `q`/`e` nudge camera yaw, `m` toggle mouse look
- `space` pause/resume
- `wasd` move the player (streaming focus follows)
The demo launches fullscreen with mouse capture enabled by default.
The mouse spawn uses a ray/plane intersection at the configured spawn height.
The overlay displays FPS, chunk count, and current material.

You can also launch via:

```bash
falling-sand demo
python -m falling_sand demo
```
