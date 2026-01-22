import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_pauli_sum_matrix():
    q = cirq.LineQubit.range(3)
    paulisum = cirq.X(q[0]) * cirq.X(q[1]) + cirq.Z(q[0])
    H1 = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, 0.0], [1.0, 0.0, 0.0, -1.0]])
    assert np.allclose(H1, paulisum.matrix())
    assert np.allclose(H1, paulisum.matrix([q[0], q[1]]))
    H2 = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, -1.0]])
    assert np.allclose(H2, paulisum.matrix([q[1], q[0]]))
    H3 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
    assert np.allclose(H3, paulisum.matrix([q[1], q[2], q[0]]))