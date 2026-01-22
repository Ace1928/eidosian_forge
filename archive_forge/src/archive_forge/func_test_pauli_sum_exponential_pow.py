import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('psum_exp, power, expected_psum', ((cirq.PauliSumExponential(cirq.Z(q1), np.pi / 2), 5, cirq.PauliSumExponential(cirq.Z(q1), 5 * np.pi / 2)), (cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Y(q2), sympy.Symbol('theta')), 5, cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Y(q2), 5 * sympy.Symbol('theta'))), (cirq.PauliSumExponential(cirq.X(q0) * cirq.Y(q1) + cirq.Y(q2) * cirq.Z(q3), np.pi), 5, cirq.PauliSumExponential(cirq.X(q0) * cirq.Y(q1) + cirq.Y(q2) * cirq.Z(q3), 5 * np.pi))))
def test_pauli_sum_exponential_pow(psum_exp, power, expected_psum):
    assert psum_exp ** power == expected_psum