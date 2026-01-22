from typing import Any, Dict, Optional, Sequence
import cirq
def test_identity_fallback_does_not_join():
    state = create_container(qs2)
    assert len(set(state.values())) == 3
    state._act_on_fallback_(cirq.I, (q0, q1))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0] is not state[None]