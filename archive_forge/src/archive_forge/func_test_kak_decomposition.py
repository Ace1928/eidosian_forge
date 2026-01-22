import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('target', [np.eye(4), SWAP, SWAP * 1j, CZ, CNOT, SWAP @ CZ] + [cirq.testing.random_unitary(4) for _ in range(10)])
def test_kak_decomposition(target):
    kak = cirq.kak_decomposition(target)
    np.testing.assert_allclose(cirq.unitary(kak), target, atol=1e-08)