# Eidosian PyParticles

A high-performance, strictly typed, and modular Python rewrite of the [Haskell Particle Life](https://github.com/Ace1928/pyparticles) simulation.

## 🌟 Overview

This project implements a particle system with emergent behavior based on simple pairwise interaction rules. It features:
- **NumPy Vectorization**: Optimized O(N²) interactions using efficient array broadcasting.
- **Eidosian Architecture**: Modular design separating Simulation, Rendering, and Configuration.
- **Pygame Visualization**: Smooth, real-time rendering.
- **Configurable Rules**: Easy-to-tune attraction/repulsion matrices.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NumPy
- Pygame

### Installation

```bash
cd game_forge/pyparticles
pip install -r requirements.txt
```

### Running

```bash
# Run with default settings (N=500)
python3 run.py

# Run with custom particle count
python3 run.py --num 800

# Run headless benchmark
python3 run.py --no-render
```

## 🏗 Architecture

- **`src/pyparticles/simulation.py`**: The core physics engine. Uses NumPy for vectorized force calculations.
- **`src/pyparticles/renderer.py`**: Visualization layer using Pygame. Handles coordinate mapping and efficient drawing.
- **`src/pyparticles/config.py`**: Central configuration for physics constants and interaction rules.
- **`src/pyparticles/main.py`**: Application entry point and loop orchestration.

## 📊 Performance

The simulation uses an O(N²) algorithm for particle interactions.
- **N=500**: ~60+ FPS (Smooth)
- **N=1000**: ~15 FPS (CPU Bound)

For N > 1000, consider implementing a spatial partition grid (Quadtree/Hash) or using GPU acceleration (Numba/CUDA).

## 🧪 Testing

Run the test suite:

```bash
make test
```

## 📜 License

MIT License. Based on the original work by Mitchell Vitez.
