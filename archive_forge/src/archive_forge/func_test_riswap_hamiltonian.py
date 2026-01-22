import numpy as np
import pytest
import sympy
from scipy import linalg
import cirq
@pytest.mark.parametrize('angle_rads', (-2 * np.pi / 3, -0.2, 0.4, np.pi / 4))
def test_riswap_hamiltonian(angle_rads):
    actual = cirq.unitary(cirq.riswap(angle_rads))
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    xx = np.kron(x, x)
    yy = np.kron(y, y)
    expected = linalg.expm(+0.5j * angle_rads * (xx + yy))
    assert np.allclose(actual, expected)