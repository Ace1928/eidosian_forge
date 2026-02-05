# Game Forge Installation

Game Forge is a collection of independent simulations and experiments. There is no
single unified Python API yet.

## Quick Start

```bash
source /home/lloyd/eidosian_forge/eidosian_venv/bin/activate
pip install -e ./game_forge
python game_forge/tools/run.py --list
```

## Run A Target

```bash
source /home/lloyd/eidosian_forge/eidosian_venv/bin/activate
python game_forge/tools/run.py gene-particles
python game_forge/tools/run.py algorithms-lab-demo -- --algorithm sph --visual
```

## Dependencies

- numpy (core)
- pygame (visual renderers)
- numba, scipy (algorithms_lab)
- python-chess (agentic_chess)
- cupy, pyopencl (optional GPU backends)

## Notes

- Many modules import `eidosian_core`, which lives in `lib/eidosian_core` at the repo root.
- The launcher adds `lib` to `PYTHONPATH` automatically.
- Some simulations are experimental and may require additional system libraries.
