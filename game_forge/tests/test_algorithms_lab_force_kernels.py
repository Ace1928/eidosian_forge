from __future__ import annotations

import numpy as np
import pytest

from algorithms_lab.backends import HAS_NUMBA
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.forces.base import ForceDefinition, ForceType
from algorithms_lab.forces.kernels import accumulate_from_pack
from algorithms_lab.forces.registry import ForceRegistry


pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="numba is required for force kernels")


def _empty_pack(num_types: int = 1):
    registry = ForceRegistry(num_types=num_types, forces=[], _skip_defaults=True)
    return registry.pack()


def test_accumulate_from_pack_validates_positions_shape() -> None:
    pack = _empty_pack()
    positions = np.zeros((4,), dtype=np.float32)
    type_ids = np.zeros((1,), dtype=np.int32)
    rows = np.zeros((0,), dtype=np.int32)
    cols = np.zeros((0,), dtype=np.int32)
    domain = Domain(mins=np.zeros(2), maxs=np.ones(2), wrap=WrapMode.CLAMP)

    with pytest.raises(ValueError, match="positions must be of shape"):
        accumulate_from_pack(positions, type_ids, rows, cols, pack, domain)


def test_accumulate_from_pack_validates_edge_lengths() -> None:
    pack = _empty_pack()
    positions = np.zeros((2, 2), dtype=np.float32)
    type_ids = np.zeros((2,), dtype=np.int32)
    rows = np.array([0], dtype=np.int32)
    cols = np.array([1, 0], dtype=np.int32)
    domain = Domain(mins=np.zeros(2), maxs=np.ones(2), wrap=WrapMode.CLAMP)

    with pytest.raises(ValueError, match="rows/cols must have the same length"):
        accumulate_from_pack(positions, type_ids, rows, cols, pack, domain)


def test_accumulate_from_pack_requires_masses_for_mass_weighted() -> None:
    registry = ForceRegistry(num_types=2)
    registry.enable_force("Gravity", True)
    pack = registry.pack()
    positions = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    type_ids = np.array([0, 1], dtype=np.int32)
    rows = np.array([0, 1], dtype=np.int32)
    cols = np.array([1, 0], dtype=np.int32)
    domain = Domain(mins=np.zeros(2), maxs=np.ones(2), wrap=WrapMode.CLAMP)

    with pytest.raises(ValueError, match="masses must be provided"):
        accumulate_from_pack(positions, type_ids, rows, cols, pack, domain)


def test_accumulate_from_pack_linear_force_symmetry() -> None:
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    linear = ForceDefinition(
        name="Linear",
        force_type=ForceType.LINEAR,
        matrix=matrix,
        min_radius=0.0,
        max_radius=1.0,
        strength=1.0,
        params=np.zeros(4, dtype=np.float32),
    )
    registry = ForceRegistry(num_types=2, forces=[], _skip_defaults=True)
    registry.add_force(linear)
    pack = registry.pack()

    positions = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    type_ids = np.array([0, 1], dtype=np.int32)
    rows = np.array([0, 1], dtype=np.int32)
    cols = np.array([1, 0], dtype=np.int32)
    domain = Domain(mins=np.zeros(2), maxs=np.ones(2), wrap=WrapMode.CLAMP)

    acc = accumulate_from_pack(positions, type_ids, rows, cols, pack, domain)
    assert acc.shape == positions.shape
    assert acc[0, 0] == pytest.approx(-1.0, rel=1e-4, abs=1e-4)
    assert acc[1, 0] == pytest.approx(1.0, rel=1e-4, abs=1e-4)
