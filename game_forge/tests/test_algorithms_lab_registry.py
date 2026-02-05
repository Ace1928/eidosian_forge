from __future__ import annotations

import numpy as np

from algorithms_lab.forces.registry import ForceRegistry


def test_registry_set_num_types_resizes_matrices() -> None:
    np.random.seed(0)
    registry = ForceRegistry(num_types=2)
    before = registry.forces[0].matrix.copy()
    registry.set_num_types(3)
    after = registry.forces[0].matrix
    assert after.shape == (3, 3)
    assert np.allclose(after[:2, :2], before)


def test_registry_round_trip() -> None:
    registry = ForceRegistry(num_types=3)
    payload = registry.to_dict()
    restored = ForceRegistry.from_dict(payload)
    assert restored.num_types == registry.num_types
    assert len(restored.forces) == len(registry.forces)
    for original, restored_force in zip(registry.forces, restored.forces, strict=True):
        assert original.name == restored_force.name
        assert original.matrix.shape == restored_force.matrix.shape


def test_registry_clear_all_zeros() -> None:
    registry = ForceRegistry(num_types=2)
    registry.randomize_all()
    registry.clear_all()
    for force in registry.forces:
        assert float(force.matrix.sum()) == 0.0
