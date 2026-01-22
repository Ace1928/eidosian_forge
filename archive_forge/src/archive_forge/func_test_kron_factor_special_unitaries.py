import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('f1,f2', [(cirq.testing.random_special_unitary(2), cirq.testing.random_special_unitary(2)) for _ in range(10)])
def test_kron_factor_special_unitaries(f1, f2):
    p = cirq.kron(f1, f2)
    g, g1, g2 = cirq.kron_factor_4x4_to_2x2s(p)
    assert np.allclose(cirq.kron(g1, g2), p)
    assert abs(g - 1) < 1e-06
    assert cirq.is_special_unitary(g1)
    assert cirq.is_special_unitary(g2)
    assert_kronecker_factorization_within_tolerance(p, g, g1, g2)