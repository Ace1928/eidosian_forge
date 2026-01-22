import numpy as np
import cirq
import pytest
def test_approx_equality_of_gates():
    state = np.array([1, 0, 0, 0], dtype=np.complex64)
    gate_1 = cirq.StatePreparationChannel(state)
    gate_2 = cirq.StatePreparationChannel(state)
    assert cirq.approx_eq(gate_1, gate_2), 'Equal state not leading to same gate'
    assert not cirq.approx_eq(gate_1, state), 'Different object types cannot be approx equal'
    perturbed_state = np.array([1 - 1e-09, 1e-10, 0, 0], dtype=np.complex64)
    gate_3 = cirq.StatePreparationChannel(perturbed_state)
    assert cirq.approx_eq(gate_3, gate_1), 'Almost equal states should lead to the same gate'
    different_state = np.array([1 - 1e-05, 0.0001, 0, 0], dtype=np.complex64)
    gate_4 = cirq.StatePreparationChannel(different_state)
    assert not cirq.approx_eq(gate_4, gate_1), 'Different states should not lead to the same gate'
    assert cirq.approx_eq(gate_4, gate_1, atol=0.001), "Gates with difference in states under the tolerance aren't equal"
    assert not cirq.approx_eq(gate_4, gate_1, atol=1e-06), 'Gates with difference in states over the tolerance are equal'