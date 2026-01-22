import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('split', [True, False])
def test_simulate_random_unitary(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype, split_untangled_states=split)
    for _ in range(10):
        random_circuit = cirq.testing.random_circuit(qubits=[q0, q1], n_moments=8, op_density=0.99)
        circuit_unitary = []
        for x in range(4):
            result = simulator.simulate(random_circuit, qubit_order=[q0, q1], initial_state=x)
            circuit_unitary.append(result.final_state_vector)
        np.testing.assert_almost_equal(np.transpose(np.array(circuit_unitary)), random_circuit.unitary(qubit_order=[q0, q1]), decimal=6)