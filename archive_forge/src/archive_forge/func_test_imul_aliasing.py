import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_imul_aliasing():
    q0, q1, q2 = cirq.LineQubit.range(3)
    psum1 = cirq.X(q0) + cirq.Y(q1)
    psum2 = psum1
    psum2 *= cirq.X(q0) * cirq.Y(q2)
    assert psum1 is psum2
    assert psum1 == psum2