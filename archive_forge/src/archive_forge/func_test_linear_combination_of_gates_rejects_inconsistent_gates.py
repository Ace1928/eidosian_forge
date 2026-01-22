import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms', ({cirq.X: -2, cirq.CZ: 2}, {cirq.X: 1, cirq.YY: 1j, cirq.ZZ: -1}, {cirq.TOFFOLI: 0.5j, cirq.S: 0.5}))
def test_linear_combination_of_gates_rejects_inconsistent_gates(terms):
    with pytest.raises(ValueError):
        cirq.LinearCombinationOfGates(terms)
    combination = cirq.LinearCombinationOfGates({})
    with pytest.raises(ValueError):
        combination.update(terms)
    combination = cirq.LinearCombinationOfGates({})
    with pytest.raises(ValueError):
        for gate, coefficient in terms.items():
            combination[gate] += coefficient