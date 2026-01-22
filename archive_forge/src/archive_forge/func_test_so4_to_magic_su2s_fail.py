import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('mat', [np.diag([0, 1, 1, 1]), np.diag([0.5, 2, 1, 1]), np.diag([1, 1j, 1, 1]), np.diag([1, 1, 1, -1])])
def test_so4_to_magic_su2s_fail(mat):
    with pytest.raises(ValueError):
        _ = cirq.so4_to_magic_su2s(mat)