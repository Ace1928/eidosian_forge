import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def assert_linear_combinations_are_equal(actual: Union[cirq.LinearCombinationOfGates, cirq.LinearCombinationOfOperations], expected: Union[cirq.LinearCombinationOfGates, cirq.LinearCombinationOfOperations]) -> None:
    if not actual and (not expected):
        assert len(actual) == 0
        assert len(expected) == 0
        return
    actual_matrix = get_matrix(actual)
    expected_matrix = get_matrix(expected)
    assert np.allclose(actual_matrix, expected_matrix)
    actual_expansion = cirq.pauli_expansion(actual)
    expected_expansion = cirq.pauli_expansion(expected)
    assert set(actual_expansion.keys()) == set(expected_expansion.keys())
    for name in actual_expansion.keys():
        assert abs(actual_expansion[name] - expected_expansion[name]) < 1e-12