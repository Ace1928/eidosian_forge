import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_sparse_simulator_repr():
    qubits = cirq.LineQubit.range(2)
    args = cirq.StateVectorSimulationState(available_buffer=np.array([0, 1, 0, 0]).reshape((2, 2)), prng=cirq.value.parse_random_state(0), qubits=qubits, initial_state=np.array([0, 1, 0, 0], dtype=np.complex64).reshape((2, 2)), dtype=np.complex64)
    step = cirq.SparseSimulatorStep(sim_state=args, dtype=np.complex64)
    assert repr(step) == "cirq.SparseSimulatorStep(sim_state=cirq.StateVectorSimulationState(initial_state=np.array([[0j, (1+0j)], [0j, 0j]], dtype=np.dtype('complex64')), qubits=(cirq.LineQubit(0), cirq.LineQubit(1)), classical_data=cirq.ClassicalDataDictionaryStore()), dtype=np.dtype('complex64'))"