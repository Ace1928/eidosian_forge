import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_pauli_sum_repr():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.Y(q[0]) * cirq.Y(q[1])
    psum = pstr1 + 2 * pstr2 + 1
    cirq.testing.assert_equivalent_repr(psum)