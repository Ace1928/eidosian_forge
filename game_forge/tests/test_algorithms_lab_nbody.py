import numpy as np

from algorithms_lab.barnes_hut import BarnesHutTree
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.fmm2d import FMM2D
from algorithms_lab.metrics import direct_nbody_acceleration, l2_relative_error


def test_barnes_hut_matches_direct_reasonably() -> None:
    rng = np.random.default_rng(3)
    positions = rng.random((32, 2), dtype=np.float32)
    masses = np.ones(32, dtype=np.float32)
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.NONE)
    reference = direct_nbody_acceleration(positions, masses, domain)
    tree = BarnesHutTree(domain)
    estimate = tree.compute_acceleration(positions, masses, theta=0.3)
    error = l2_relative_error(reference, estimate)
    assert error < 0.35


def test_fmm2d_matches_direct_reasonably() -> None:
    rng = np.random.default_rng(5)
    positions = rng.random((32, 2), dtype=np.float32)
    masses = np.ones(32, dtype=np.float32)
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.NONE)
    reference = direct_nbody_acceleration(positions, masses, domain)
    fmm = FMM2D(domain, cell_size=0.08)
    estimate = fmm.compute_acceleration(positions, masses)
    error = l2_relative_error(reference, estimate)
    assert error < 0.2
