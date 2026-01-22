import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_matrix_to_gates_tolerance_half_turn_phasing():
    a = np.pi / 2 + 0.01
    c, s = (np.cos(a), np.sin(a))
    nearly_x = np.array([[c, -s], [s, c]])
    z1 = np.diag([1, np.exp(1j * 1.2)])
    z2 = np.diag([1, np.exp(1j * 1.6)])
    phased_nearly_x = z1.dot(nearly_x).dot(z2)
    optimized_away = cirq.single_qubit_matrix_to_gates(phased_nearly_x, tolerance=0.1)
    assert len(optimized_away) == 2
    kept = cirq.single_qubit_matrix_to_gates(phased_nearly_x, tolerance=0.0001)
    assert len(kept) == 3