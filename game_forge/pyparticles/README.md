# 💎 Eidosian PyParticles V6.2

> *"Precision is the foundation of elegance; emergence is the path to understanding."*

A **production-grade, ultra-high-performance particle physics simulation** with advanced physics, wave mechanics, quantum-inspired exclusion, and scalable architecture.

## ✨ Features

### Physics Engine
- **10 Force Types**: Linear, Inverse, Inverse-Square, Inverse-Cube, Yukawa, Lennard-Jones, Morse, Gaussian, Exponential, Repel-Only
- **Wave Mechanics**: Dynamic wave perimeters with crest/trough/zero-crossing effects
- **Exclusion Mechanics**: Pauli-like repulsion with fermionic/bosonic behaviors
- **Spin Dynamics**: Quantum-inspired spin states with stochastic flip mechanics
- **Velocity Verlet Integration**: Symplectic, energy-conserving integrator
- **Berendsen Thermostat**: Temperature control for NVT ensemble

### Rendering
- **ModernGL OpenGL**: Hardware-accelerated GPU rendering
- **Advanced Shaders**: Wave visualization, energy effects, LOD
- **Multiple Render Modes**: Standard, Wave, Energy, Minimal

### Performance
- **Numba JIT**: All physics kernels compiled to machine code
- **Spatial Hashing**: O(N) neighbor lookup via grid
- **Morton Encoding**: Cache-optimized memory access patterns
- **Parallel Execution**: Multi-threaded force computation

### GUI
- **Real-time Controls**: Particle size, world size, velocity limits
- **Simulation Presets**: Small, Large, Huge, Classic, Emergence
- **Performance Monitoring**: FPS, physics time, render time
- **Species Editor**: Randomize and configure particle types

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenGL 3.3+ capable GPU

### Installation

```bash
cd game_forge/pyparticles
pip install -r requirements.txt
```

### Running

```bash
# Default (10,000 particles, 16 species)
./bin/pyparticles_app

# Custom configuration
./bin/pyparticles_app --num 20000 --types 8 --world-size 200.0

# With preset
./bin/pyparticles_app --preset huge

# Headless benchmark
PYTHONPATH=src SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m pyparticles.app --benchmark

# Headless profiling
PYTHONPATH=src SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m pyparticles.app --profile -n 5000

# Render benchmark (requires OpenGL context; skipped under dummy driver)
PYTHONPATH=src python -m pyparticles.app --benchmark

# Compare to baseline
PYTHONPATH=src python - <<'PY'
from pyparticles.profiling import Benchmarker
from pyparticles.core.types import SimulationConfig
bench = Benchmarker()
bench.benchmark_physics(SimulationConfig.default(), n_iterations=10)
print(bench.compare_to_baseline("benchmarks/baseline.json"))
PY
```

## 🏗️ Architecture

```
pyparticles/
├── src/pyparticles/
│   ├── app.py                    # Main application loop
│   ├── core/
│   │   └── types.py              # SimulationConfig, ParticleState, SpeciesConfig
│   ├── physics/
│   │   ├── engine.py             # PhysicsEngine orchestrator
│   │   ├── kernels.py            # Numba JIT physics kernels
│   │   ├── forces/               # Pluggable force types
│   │   │   ├── base.py           # Force interface
│   │   │   ├── potentials.py     # All 10 force implementations
│   │   │   └── registry.py       # Force registry with presets
│   │   ├── wave/                 # Wave mechanics
│   │   │   ├── types.py          # WaveConfig, WaveProfile
│   │   │   ├── kernels.py        # Wave computation kernels
│   │   │   ├── analyzer.py       # Wave statistics
│   │   │   └── registry.py       # Wave presets
│   │   ├── exclusion/            # Quantum-inspired exclusion
│   │   │   ├── types.py          # SpinState, ParticleBehavior
│   │   │   ├── kernels.py        # Exclusion/spin kernels
│   │   │   └── registry.py       # Exclusion presets
│   │   └── spatial/              # Spatial optimization
│   │       └── __init__.py       # Morton encoding, adaptive grid
│   ├── rendering/
│   │   ├── gl_renderer.py        # ModernGL renderer
│   │   └── shaders/              # GLSL shaders
│   ├── ui/
│   │   └── gui_v2.py             # Collapsible GUI controls
│   └── profiling/
│       └── __init__.py           # Benchmarking tools
└── tests/                        # 132+ comprehensive tests
```

## 🔬 Physics

### Force Types

| Type | Formula | Use Case |
|------|---------|----------|
| Linear | `F = k(1 - r/r_max)` | Particle Life |
| Inverse | `F = k/r` | Magnetic-like |
| Inverse Square | `F = k/r^2` | Gravity/Coulomb |
| Inverse Cube | `F = k/r^3` | Dipole-dipole |
| Yukawa | `F = k*exp(-r/lambda)/r` | Screened Coulomb |
| Lennard-Jones | `F = k[(sigma/r)^12 - (sigma/r)^6]` | Molecular |
| Morse | `F = k*exp(-2a(r-r0)) - exp(-a(r-r0))` | Bond-like |
| Gaussian | `F = k*exp(-r^2/2*sigma^2)` | Soft localized |
| Exponential | `F = k*exp(-r/lambda)` | Short-range decay |
| Repel Only | `F = max(0, k/r^2)` | Exclusion zone |

### Wave Mechanics

Each particle has a dynamic wave perimeter:
- **Crest**: Maximum protrusion → enhanced attraction
- **Trough**: Maximum indentation → reduced interaction
- **Zero Crossing**: Neutral surface → normal force

Phase relationships create emergent behaviors:
- **Constructive interference**: 2× force multiplier
- **Destructive interference**: 0.5× force multiplier
- **Standing waves**: Phase-locked oscillations

### Exclusion Mechanics

Quantum-inspired particle behaviors:
- **Fermionic**: Same-spin particles strongly repel (Pauli-like)
- **Bosonic**: Particles can occupy same state, slight attraction
- **Classical**: No exclusion effects

Spin dynamics:
- Spin states: UP (+1), DOWN (-1), NONE (0)
- Stochastic spin flips based on kinetic energy
- Spin-spin correlation for emergent magnetization

## 📊 Performance

| Particles | ms/step | Notes |
|-----------|---------|-------|
| 1,000 | ~7 | Small world preset |
| 5,000 | ~199 | Default preset |
| 10,000 | ~800 | Default preset |

These numbers are a baseline snapshot (Feb 5, 2026) on the current dev machine; expect variation across hardware.

Target: 100k+ particles with GPU compute shaders (planned).

### Troubleshooting Numba Parallel Warnings
- If you see warnings about `/dev/shm` permissions, ensure `/dev/shm` is writable.
  On Linux that typically means the tmpfs mount is present and writable.
  As a workaround, you can set `NUMBA_THREADING_LAYER=workqueue` before running.

## 🧪 Testing

```bash
# Run all tests (132+)
PYTHONPATH=src pytest tests/ -v

# Run specific module
PYTHONPATH=src pytest tests/test_forces.py -v

# With coverage
PYTHONPATH=src pytest tests/ --cov=pyparticles --cov-report=html
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| Space | Pause/Resume |
| R | Reset simulation |
| Tab | Toggle GUI |
| +/- | Adjust particle count |
| Mouse wheel | Zoom |

## 📜 License

MIT License. Inspired by [Particle Life](https://github.com/hunar4321/particle-life) and the original Haskell implementation.

---

*"The particle is the universe in miniature; the simulation, the universe in understanding."*
