import numpy as np
import pytest

from algorithms_lab.backends import HAS_NUMBA
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.forces import ForceDefinition, ForceRegistry, ForceType, accumulate_from_registry


def test_force_registry_pack_shapes():
    registry = ForceRegistry(num_types=3, forces=[], _skip_defaults=True)
    matrix = np.array(
        [
            [0.0, 1.0, -1.0],
            [1.0, 0.0, 0.5],
            [-1.0, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    force = ForceDefinition(
        name="Inverse",
        force_type=ForceType.INVERSE,
        matrix=matrix,
        min_radius=0.0,
        max_radius=1.0,
        strength=1.0,
        params=np.array([0.01], dtype=np.float32),
    )
    registry.add_force(force)
    pack = registry.pack()
    assert pack.matrices.shape == (1, 3, 3)
    assert pack.force_types.shape == (1,)
    assert pack.params.shape == (1, 4)
    assert pack.mass_weighted.shape == (1,)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba is required for force kernels")
def test_force_kernel_symmetry():
    registry = ForceRegistry(num_types=2, forces=[], _skip_defaults=True)
    matrix = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    force = ForceDefinition(
        name="Inverse",
        force_type=ForceType.INVERSE,
        matrix=matrix,
        min_radius=0.0,
        max_radius=1.0,
        strength=1.0,
        params=np.array([0.0], dtype=np.float32),
    )
    registry.add_force(force)

    positions = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
        ],
        dtype=np.float32,
    )
    type_ids = np.array([0, 1], dtype=np.int32)
    rows = np.array([0, 1], dtype=np.int32)
    cols = np.array([1, 0], dtype=np.int32)
    domain = Domain(
        mins=np.array([0.0, 0.0], dtype=np.float32),
        maxs=np.array([1.0, 1.0], dtype=np.float32),
        wrap=WrapMode.NONE,
    )

    acc = accumulate_from_registry(positions, type_ids, rows, cols, registry, domain)
    assert acc.shape == (2, 2)
    assert np.isfinite(acc).all()
    assert np.allclose(acc[0, 0], -acc[1, 0], rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba is required for force kernels")
def test_force_kernel_mass_weighted():
    registry = ForceRegistry(num_types=2, forces=[], _skip_defaults=True)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    force = ForceDefinition(
        name="Gravity",
        force_type=ForceType.INVERSE_CUBE,
        matrix=matrix,
        min_radius=0.0,
        max_radius=1.0,
        strength=1.0,
        params=np.array([0.0], dtype=np.float32),
        mass_weighted=True,
    )
    registry.add_force(force)

    positions = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)
    type_ids = np.array([0, 1], dtype=np.int32)
    rows = np.array([0, 1], dtype=np.int32)
    cols = np.array([1, 0], dtype=np.int32)
    masses = np.array([1.0, 2.0], dtype=np.float32)
    domain = Domain(
        mins=np.array([0.0, 0.0], dtype=np.float32),
        maxs=np.array([1.0, 1.0], dtype=np.float32),
        wrap=WrapMode.NONE,
    )

    acc = accumulate_from_registry(positions, type_ids, rows, cols, registry, domain, masses=masses)
    assert acc.shape == (2, 2)
    assert np.isfinite(acc).all()
    assert abs(acc[0, 0]) > 0.0
