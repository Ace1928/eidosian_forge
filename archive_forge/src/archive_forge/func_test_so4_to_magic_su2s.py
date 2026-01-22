import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('m', [cirq.testing.random_special_orthogonal(4) for _ in range(10)])
def test_so4_to_magic_su2s(m):
    a, b = cirq.so4_to_magic_su2s(m)
    m2 = recompose_so4(a, b)
    assert_magic_su2_within_tolerance(m2, a, b)
    assert np.allclose(m, m2)