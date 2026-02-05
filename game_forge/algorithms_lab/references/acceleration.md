# Acceleration References

- Numba provides a JIT compiler for Python/NumPy to accelerate loops.
- SciPy's cKDTree supports fast neighbor searches and periodic boundaries
  via the `boxsize` option.
- CuPy provides a NumPy-compatible API for CUDA/ROCm GPU arrays.
- PyOpenCL provides access to OpenCL devices (including integrated GPUs).

Sources:
- https://numba.readthedocs.io/en/stable/release/0.59.0-notes.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
- https://docs.cupy.dev/en/stable/
- https://documen.tician.de/pyopencl/
