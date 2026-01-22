import pytest
import numpy as np
from ase.quaternions import Quaternion
def rand_rotm(rng=np.random.RandomState(0)):
    """Axis & angle rotations."""
    u = rng.rand(3)
    theta = rng.rand() * np.pi * 2
    return axang_rotm(u, theta)