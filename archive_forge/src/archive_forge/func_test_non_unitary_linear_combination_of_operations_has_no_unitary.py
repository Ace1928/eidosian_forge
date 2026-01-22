import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms', ({cirq.CNOT(q0, q1): 1.1}, {cirq.CZ(q0, q1) ** sympy.Symbol('t'): 1}, {cirq.X(q0): 1, cirq.S(q0): 1}))
def test_non_unitary_linear_combination_of_operations_has_no_unitary(terms):
    combination = cirq.LinearCombinationOfOperations(terms)
    assert not cirq.has_unitary(combination)
    with pytest.raises((TypeError, ValueError)):
        _ = cirq.unitary(combination)