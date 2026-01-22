import numpy as np
import pytest
from cirq.testing import (
from cirq.linalg import is_unitary, is_orthogonal, is_special_unitary, is_special_orthogonal
def test_random_unitary():
    u1 = random_unitary(2)
    u2 = random_unitary(2)
    assert is_unitary(u1)
    assert is_unitary(u2)
    assert not np.allclose(u1, u2)