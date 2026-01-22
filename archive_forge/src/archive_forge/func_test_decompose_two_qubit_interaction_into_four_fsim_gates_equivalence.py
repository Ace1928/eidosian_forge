import itertools
import random
from typing import Any
import numpy as np
import pytest
import sympy
import cirq
from cirq.transformers.analytical_decompositions.two_qubit_to_fsim import (
@pytest.mark.parametrize('obj,fsim_gate', itertools.product(UNITARY_OBJS, FEASIBLE_FSIM_GATES))
def test_decompose_two_qubit_interaction_into_four_fsim_gates_equivalence(obj: Any, fsim_gate: cirq.FSimGate):
    qubits = obj.qubits if isinstance(obj, cirq.Operation) else cirq.LineQubit.range(2)
    circuit = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(obj, fsim_gate=fsim_gate)
    desired_unitary = obj if isinstance(obj, np.ndarray) else cirq.unitary(obj)
    for operation in circuit.all_operations():
        assert len(operation.qubits) < 2 or operation.gate == fsim_gate
    assert len(circuit) <= 4 * 3 + 5
    assert cirq.approx_eq(circuit.unitary(qubit_order=qubits), desired_unitary, atol=0.0001)