import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_density_matrix_is_not_last_object():
    sim = cirq.DensityMatrixSimulator()
    q = cirq.LineQubit(0)
    initial_state = np.array([[1, 0], [0, 0]], dtype=np.complex64)
    circuit = cirq.Circuit(cirq.wait(q))
    result = sim.simulate(circuit, initial_state=initial_state)
    assert result.final_density_matrix is not initial_state
    assert not np.shares_memory(result.final_density_matrix, initial_state)
    np.testing.assert_equal(result.final_density_matrix, initial_state)