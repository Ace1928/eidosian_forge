import numpy as np
import pytest
import cirq
def test_gate_with_act_on():

    class CustomGate(cirq.testing.SingleQubitGate):

        def _act_on_(self, sim_state, qubits):
            if isinstance(sim_state, cirq.StabilizerChFormSimulationState):
                qubit = sim_state.qubit_map[qubits[0]]
                sim_state.state.gamma[qubit] += 1
                return True
    state = cirq.StabilizerStateChForm(num_qubits=3)
    args = cirq.StabilizerChFormSimulationState(qubits=cirq.LineQubit.range(3), prng=np.random.RandomState(), initial_state=state)
    cirq.act_on(CustomGate(), args, [cirq.LineQubit(1)])
    np.testing.assert_allclose(state.gamma, [0, 1, 0])