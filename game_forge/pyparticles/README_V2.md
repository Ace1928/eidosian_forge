# Eidosian PyParticles V2

The high-performance, modular Python implementation of Particle Life.

## 🏗 Architecture (The "Forge" Pattern)

This V2 refactor introduces a strictly modular architecture designed for extensibility and performance.

### 🧩 Modules

*   **`pyparticles.core`**: Type definitions, Dataclasses, and Enums.
*   **`pyparticles.physics`**:
    *   `kernels.py`: Pure Numba JIT functions for force calculation and integration.
    *   `engine.py`: Manages memory (SoA state) and orchestration.
*   **`pyparticles.rendering`**:
    *   `canvas.py`: Pygame-based renderer with support for Sprites, Pixels, and Glow modes.
*   **`pyparticles.ui`**:
    *   `gui.py`: `pygame_gui` integration for real-time parameter tuning.
*   **`pyparticles.app`**: The application loop and dependency injection.

## 🚀 Performance

*   **Spatial Hashing**: O(N) neighbor search using a grid.
*   **Numba JIT**: Python code compiled to machine code for C++ level speeds.
*   **Structure of Arrays (SoA)**: Contiguous memory blocks for positions and velocities, maximizing cache locality.
*   **Render Modes**:
    *   *Sprites*: Standard blit (Good for N < 5,000).
    *   *Pixels*: Direct frame buffer manipulation (Good for N > 10,000).

## 🛠 Usage

```bash
# Standard Run
python3 bin/pyparticles_app

# High Performance Mode
python3 bin/pyparticles_app --num 10000 --mode pixels

# Glow Mode (Visual Candy)
python3 bin/pyparticles_app --num 2000 --mode glow
```

## 🧪 Testing

```bash
export PYTHONPATH=src
pytest tests/
```
