import numpy as np
import pytest

from algorithms_lab.barnes_hut import BarnesHutTree
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.metrics import direct_nbody_acceleration, l2_relative_error

numba = pytest.importorskip("numba", reason="numba required")


def test_barnes_hut_numba_matches_numpy() -> None:
    rng = np.random.default_rng(11)
    positions = rng.random((64, 2), dtype=np.float32)
    masses = np.ones(64, dtype=np.float32)
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)
    tree = BarnesHutTree(domain)
    acc_numpy = tree.compute_acceleration(positions, masses, theta=0.4, backend="numpy")
    acc_numba = tree.compute_acceleration(positions, masses, theta=0.4, backend="numba")
    error = l2_relative_error(acc_numpy, acc_numba)
    assert error < 1e-3


def test_barnes_hut_numba_reasonable_against_direct() -> None:
    rng = np.random.default_rng(12)
    positions = rng.random((48, 2), dtype=np.float32)
    masses = np.ones(48, dtype=np.float32)
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)
    reference = direct_nbody_acceleration(positions, masses, domain)
    tree = BarnesHutTree(domain)
    estimate = tree.compute_acceleration(positions, masses, theta=0.4, backend="numba")
    error = l2_relative_error(reference, estimate)
    assert error < 0.4
