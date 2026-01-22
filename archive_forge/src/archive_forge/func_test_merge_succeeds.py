from typing import Any, Dict, Optional, Sequence
import cirq
def test_merge_succeeds():
    state = create_container(qs2, False)
    merged = state.create_merged_state()
    assert merged.qubits == (q0, q1)