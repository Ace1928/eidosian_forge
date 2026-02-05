import numpy as np

from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.fmm_multilevel import MultiLevelFMM
from algorithms_lab.metrics import direct_nbody_acceleration, l2_relative_error


def test_multilevel_fmm_reasonable_accuracy() -> None:
    rng = np.random.default_rng(21)
    positions = rng.random((64, 2), dtype=np.float32)
    masses = np.ones(64, dtype=np.float32)
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.NONE)
    reference = direct_nbody_acceleration(positions, masses, domain)
    fmm = MultiLevelFMM(domain, levels=4)
    estimate = fmm.compute_acceleration(positions, masses)
    error = l2_relative_error(reference, estimate)
    assert error < 0.6
