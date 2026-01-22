import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, expected_expansion', (({}, {}), ({cirq.X(q0): -10, cirq.Y(q0): 20}, {'X': -10, 'Y': 20}), ({cirq.X(q0): -10, cirq.Y(q1): 20}, {'XI': -10, 'IY': 20}), ({cirq.Y(q0): np.sqrt(0.5), cirq.H(q0): 1}, {'X': np.sqrt(0.5), 'Y': np.sqrt(0.5), 'Z': np.sqrt(0.5)}), ({cirq.Y(q0): np.sqrt(0.5), cirq.H(q2): 1}, {'IX': np.sqrt(0.5), 'YI': np.sqrt(0.5), 'IZ': np.sqrt(0.5)}), ({cirq.XX(q0, q1): -2, cirq.YY(q0, q1): 3j, cirq.ZZ(q0, q1): 4}, {'XX': -2, 'YY': 3j, 'ZZ': 4}), ({cirq.XX(q0, q1): -2, cirq.YY(q0, q2): 3j, cirq.ZZ(q1, q2): 4}, {'XXI': -2, 'YIY': 3j, 'IZZ': 4}), ({cirq.IdentityGate(2).on(q0, q3): -1, cirq.CZ(q1, q2): 2}, {'IIZI': 1, 'IZII': 1, 'IZZI': -1}), ({cirq.CNOT(q0, q1): 2, cirq.Z(q0): -1, cirq.X(q1): -1}, {'II': 1, 'ZX': -1})))
def test_linear_combination_of_operations_has_correct_pauli_expansion(terms, expected_expansion):
    combination = cirq.LinearCombinationOfOperations(terms)
    actual_expansion = cirq.pauli_expansion(combination)
    assert set(actual_expansion.keys()) == set(expected_expansion.keys())
    for name in actual_expansion.keys():
        assert abs(actual_expansion[name] - expected_expansion[name]) < 1e-12