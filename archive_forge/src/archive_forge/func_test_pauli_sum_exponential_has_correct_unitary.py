import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('psum_exp, expected_unitary', ((cirq.PauliSumExponential(cirq.X(q0), np.pi / 2), np.array([[0, 1j], [1j, 0]])), (cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Z(q1), np.pi / 2), np.array([[1j, 0, 0, 0], [0, -1j, 0, 0], [0, 0, 1j, 0], [0, 0, 0, -1j]]))))
def test_pauli_sum_exponential_has_correct_unitary(psum_exp, expected_unitary):
    assert cirq.has_unitary(psum_exp)
    assert np.allclose(cirq.unitary(psum_exp), expected_unitary)