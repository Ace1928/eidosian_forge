import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_density_matrix_step_result_repr():
    q0 = cirq.LineQubit(0)
    assert repr(cirq.DensityMatrixStepResult(sim_state=cirq.DensityMatrixSimulationState(initial_state=np.ones((2, 2)) * 0.5, qubits=[q0]))) == "cirq.DensityMatrixStepResult(sim_state=cirq.DensityMatrixSimulationState(initial_state=np.array([[(0.5+0j), (0.5+0j)], [(0.5+0j), (0.5+0j)]], dtype=np.dtype('complex64')), qubits=(cirq.LineQubit(0),), classical_data=cirq.ClassicalDataDictionaryStore()), dtype=np.dtype('complex64'))"