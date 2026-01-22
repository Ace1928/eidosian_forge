import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, expected_unitary', (({cirq.X(q0): np.sqrt(0.5), cirq.Z(q0): np.sqrt(0.5)}, np.sqrt(0.5) * np.array([[1, 1], [1, -1]])), ({cirq.IdentityGate(3).on(q0, q1, q2): np.sqrt(0.5), cirq.CCZ(q0, q1, q2): 1j * np.sqrt(0.5)}, np.diag([np.sqrt(1j), np.sqrt(1j), np.sqrt(1j), np.sqrt(1j), np.sqrt(1j), np.sqrt(1j), np.sqrt(1j), np.sqrt(-1j)]))))
def test_unitary_linear_combination_of_operations_has_correct_unitary(terms, expected_unitary):
    combination = cirq.LinearCombinationOfOperations(terms)
    assert cirq.has_unitary(combination)
    assert np.allclose(cirq.unitary(combination), expected_unitary)