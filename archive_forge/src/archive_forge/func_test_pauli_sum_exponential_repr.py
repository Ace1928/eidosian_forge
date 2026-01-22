import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('psum_exp', (cirq.PauliSumExponential(0, np.pi / 2), cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Z(q1), np.pi / 2)))
def test_pauli_sum_exponential_repr(psum_exp):
    cirq.testing.assert_equivalent_repr(psum_exp)