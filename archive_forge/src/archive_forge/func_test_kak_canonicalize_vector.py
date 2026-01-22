import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('x,y,z', [[(random.random() * 2 - 1) * np.pi * 2 for _ in range(3)] for _ in range(10)])
def test_kak_canonicalize_vector(x, y, z):
    i = np.eye(2)
    m = cirq.unitary(cirq.KakDecomposition(global_phase=1, single_qubit_operations_after=(i, i), interaction_coefficients=(x, y, z), single_qubit_operations_before=(i, i)))
    kak = cirq.kak_canonicalize_vector(x, y, z, atol=1e-10)
    a1, a0 = kak.single_qubit_operations_after
    x2, y2, z2 = kak.interaction_coefficients
    b1, b0 = kak.single_qubit_operations_before
    m2 = cirq.unitary(kak)
    assert 0.0 <= x2 <= np.pi / 4
    assert 0.0 <= y2 <= np.pi / 4
    assert -np.pi / 4 < z2 <= np.pi / 4
    assert abs(x2) >= abs(y2) >= abs(z2)
    assert x2 < np.pi / 4 - 1e-10 or z2 >= 0
    assert cirq.is_special_unitary(a1)
    assert cirq.is_special_unitary(a0)
    assert cirq.is_special_unitary(b1)
    assert cirq.is_special_unitary(b0)
    assert np.allclose(m, m2)