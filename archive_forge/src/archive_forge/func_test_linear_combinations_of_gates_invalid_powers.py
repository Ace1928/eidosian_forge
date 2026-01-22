import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, exponent', (({}, 2), ({cirq.H: 1}, 2), ({cirq.CNOT: 2}, 2), ({cirq.X: 1, cirq.S: -1}, 2), ({cirq.X: 1}, -1), ({cirq.Y: 1}, sympy.Symbol('k'))))
def test_linear_combinations_of_gates_invalid_powers(terms, exponent):
    combination = cirq.LinearCombinationOfGates(terms)
    with pytest.raises(TypeError):
        _ = combination ** exponent