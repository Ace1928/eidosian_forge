import numpy as np
import pytest

from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.gpu import CuPyNBody, OpenCLNBody, HAS_CUPY, HAS_PYOPENCL


def test_cupy_nbody_optional() -> None:
    if not HAS_CUPY:
        pytest.skip("cupy not available")
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)
    positions = np.random.rand(16, 2).astype(np.float32)
    masses = np.ones(16, dtype=np.float32)
    solver = CuPyNBody(domain, tile=8)
    acc = solver.compute_acceleration(positions, masses)
    assert acc.shape == positions.shape
    assert np.all(np.isfinite(acc))


def test_opencl_nbody_optional() -> None:
    if not HAS_PYOPENCL:
        pytest.skip("pyopencl not available")
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)
    positions = np.random.rand(16, 2).astype(np.float32)
    masses = np.ones(16, dtype=np.float32)
    solver = OpenCLNBody(domain)
    acc = solver.compute_acceleration(positions, masses)
    assert acc.shape == positions.shape
    assert np.all(np.isfinite(acc))
