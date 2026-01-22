from typing import Any, Dict, Optional, Sequence
import cirq
def test_subcircuit_identity_does_not_join():
    state = create_container(qs2)
    assert len(set(state.values())) == 3
    state.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.IdentityGate(2)(q0, q1))))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]