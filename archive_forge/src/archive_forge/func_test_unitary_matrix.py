import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_unitary_matrix():
    a, b = cirq.LineQubit.range(2)
    assert not cirq.has_unitary(2 * cirq.X(a) * cirq.Z(b))
    assert cirq.unitary(2 * cirq.X(a) * cirq.Z(b), default=None) is None
    np.testing.assert_allclose(cirq.unitary(cirq.X(a) * cirq.Z(b)), np.array([[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]]))
    np.testing.assert_allclose(cirq.unitary(1j * cirq.X(a) * cirq.Z(b)), np.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [1j, 0, 0, 0], [0, -1j, 0, 0]]))