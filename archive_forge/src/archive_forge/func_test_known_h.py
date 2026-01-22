import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def test_known_h():
    actual = cirq.single_qubit_matrix_to_gates(np.array([[1, 1], [1, -1]]) * np.sqrt(0.5), tolerance=0.001)
    assert cirq.approx_eq(actual, [cirq.Y ** (-0.5), cirq.Z], atol=1e-09)