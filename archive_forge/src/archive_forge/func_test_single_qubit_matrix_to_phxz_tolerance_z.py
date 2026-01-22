import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_matrix_to_phxz_tolerance_z():
    z = np.diag([1, np.exp(1j * 0.01)])
    optimized_away = cirq.single_qubit_matrix_to_phxz(z, atol=0.1)
    assert optimized_away is None
    kept = cirq.single_qubit_matrix_to_phxz(z, atol=0.0001)
    assert kept is not None