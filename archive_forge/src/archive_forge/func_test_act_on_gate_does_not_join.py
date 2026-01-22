from typing import Any, Dict, Optional, Sequence
import cirq
def test_act_on_gate_does_not_join():
    state = create_container(qs2)
    assert len(set(state.values())) == 3
    cirq.act_on(cirq.X, state, [q0])
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0] is not state[None]