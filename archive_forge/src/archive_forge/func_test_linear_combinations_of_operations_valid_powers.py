import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, exponent, expected_terms', (({cirq.X(q0): 1}, 2, {cirq.I(q0): 1}), ({cirq.X(q0): 1}, 3, {cirq.X(q0): 1}), ({cirq.Y(q0): 0.5}, 10, {cirq.I(q0): 2 ** (-10)}), ({cirq.Y(q0): 0.5}, 11, {cirq.Y(q0): 2 ** (-11)}), ({cirq.I(q0): 1, cirq.X(q0): 2, cirq.Y(q0): 3, cirq.Z(q0): 4}, 2, {cirq.I(q0): 30, cirq.X(q0): 4, cirq.Y(q0): 6, cirq.Z(q0): 8}), ({cirq.X(q0): 1, cirq.Y(q0): 1j}, 2, {}), ({cirq.Y(q1): 2, cirq.Z(q1): 3}, 0, {cirq.I(q1): 1})))
def test_linear_combinations_of_operations_valid_powers(terms, exponent, expected_terms):
    combination = cirq.LinearCombinationOfOperations(terms)
    actual_result = combination ** exponent
    expected_result = cirq.LinearCombinationOfOperations(expected_terms)
    assert cirq.approx_eq(actual_result, expected_result)
    assert len(actual_result) == len(expected_terms)