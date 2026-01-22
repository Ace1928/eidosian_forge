import numpy as np
import pytest
from cirq.testing import (
from cirq.linalg import is_unitary, is_orthogonal, is_special_unitary, is_special_orthogonal
def test_seeded_special_unitary():
    u1 = random_special_unitary(2, random_state=np.random.RandomState(1))
    u2 = random_special_unitary(2, random_state=np.random.RandomState(1))
    u3 = random_special_unitary(2, random_state=np.random.RandomState(2))
    assert np.allclose(u1, u2)
    assert not np.allclose(u1, u3)