from typing import Any, Dict, Optional, Sequence
import cirq
def test_reset_does_not_split_if_disabled():
    state = create_container(qs2, False)
    state.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(state.values())) == 1
    state.apply_operation(cirq.reset(q0))
    assert len(set(state.values())) == 1
    assert state[q1] is state[q0]
    assert state[None] is state[q0]