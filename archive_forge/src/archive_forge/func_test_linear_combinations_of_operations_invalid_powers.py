import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, exponent', (({}, 2), ({cirq.H(q0): 1}, 2), ({cirq.CNOT(q0, q1): 2}, 2), ({cirq.X(q0): 1, cirq.S(q0): -1}, 2), ({cirq.X(q0): 1, cirq.Y(q1): 1}, 2), ({cirq.Z(q0): 1}, -1), ({cirq.X(q0): 1}, sympy.Symbol('k'))))
def test_linear_combinations_of_operations_invalid_powers(terms, exponent):
    combination = cirq.LinearCombinationOfOperations(terms)
    with pytest.raises(TypeError):
        _ = combination ** exponent