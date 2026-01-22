import pytest
import numpy as np
from ase.quaternions import Quaternion
def test_quaternions_gimbal(rng):
    rotm = np.identity(3)
    rotm[:2, :2] *= -1
    q = Quaternion.from_matrix(rotm)
    assert not np.isnan(q.q).any()