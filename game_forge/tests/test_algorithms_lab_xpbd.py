import numpy as np

from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.xpbd import XPBFState, XPBFSolver


def test_xpbd_step_produces_valid_state() -> None:
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)
    positions = np.array([[0.2, 0.2], [0.25, 0.2], [0.8, 0.8]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    masses = np.ones(3, dtype=np.float32)
    solver = XPBFSolver(domain, h=0.2, dt=0.01, iterations=2, compliance=0.001)
    state = XPBFState(positions=positions, velocities=velocities, masses=masses)
    new_state = solver.step(state)
    assert new_state.positions.shape == positions.shape
    assert np.all(np.isfinite(new_state.positions))
    assert np.all(np.isfinite(new_state.velocities))
