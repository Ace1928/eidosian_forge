import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_pauli_sum_neg():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.Y(q[0]) * cirq.Y(q[1])
    psum1 = pstr1 + pstr2
    psum2 = -1 * pstr1 - pstr2
    assert -psum1 == psum2
    psum1 *= -1
    assert psum1 == psum2
    psum2 = psum1 * -1
    assert psum1 == -psum2