from typing import Any, Dict, Optional, Sequence
import cirq
def test_subcircuit_entanglement_causes_join_in_subset():
    state = create_container(qs3)
    assert len(set(state.values())) == 4
    state.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CNOT(q0, q1))))
    assert len(set(state.values())) == 3
    assert state[q0] is state[q1]
    state.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CNOT(q0, q2))))
    assert len(set(state.values())) == 2
    assert state[q0] is state[q1] is state[q2]