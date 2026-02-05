# GPU Backends

Algorithms Lab provides optional GPU backends for N-body acceleration.
These are designed to be useful even on integrated GPUs (via OpenCL) and
on discrete GPUs (via CUDA/ROCm through CuPy).

## Available Backends

### OpenCL (Integrated GPUs)
- `OpenCLNBody` uses PyOpenCL to run a direct N-body kernel on any
  OpenCL-capable device (including Intel iGPUs when drivers are present).
- This backend computes acceleration per particle and avoids atomics by
  giving each work item its own output slot.

### CuPy (CUDA/ROCm)
- `CuPyNBody` uses tiled CuPy operations to reduce memory pressure while
  keeping heavy math on the GPU.
- Suitable for discrete GPUs with CUDA or ROCm support.

## Quick Start

```python
import numpy as np
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.gpu import OpenCLNBody, CuPyNBody, HAS_PYOPENCL, HAS_CUPY

positions = np.random.rand(512, 2).astype(np.float32)
masses = np.ones(512, dtype=np.float32)

domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)

if HAS_PYOPENCL:
    solver = OpenCLNBody(domain)
    acc = solver.compute_acceleration(positions, masses)

if HAS_CUPY:
    solver = CuPyNBody(domain)
    acc = solver.compute_acceleration(positions, masses)
```

## Notes
- GPU kernels are currently focused on N-body acceleration; neighbor
  searches still run on CPU (NumPy/Numba/cKDTree).
- OpenCL drivers are required for the OpenCL backend.
- CuPy requires CUDA or ROCm installed and configured.
