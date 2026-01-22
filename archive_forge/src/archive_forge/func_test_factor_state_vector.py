import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
@pytest.mark.parametrize('state_1', [0, 1])
@pytest.mark.parametrize('state_2', [0, 1])
def test_factor_state_vector(state_1: int, state_2: int):
    n = 12
    for i in range(n):
        phase = np.exp(2 * np.pi * 1j * i / n)
        a = cirq.to_valid_state_vector(state_1, 1)
        b = cirq.to_valid_state_vector(state_2, 1)
        c = cirq.linalg.transformations.state_vector_kronecker_product(a, b) * phase
        a1, b1 = cirq.linalg.transformations.factor_state_vector(c, [0], validate=True)
        c1 = cirq.linalg.transformations.state_vector_kronecker_product(a1, b1)
        assert np.allclose(c, c1)
        assert np.allclose(a1, a * phase)
        assert np.allclose(b1, b)