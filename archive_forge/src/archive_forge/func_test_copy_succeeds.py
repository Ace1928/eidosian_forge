from typing import Any, Dict, Optional, Sequence
import cirq
def test_copy_succeeds():
    state = create_container(qs2, False)
    copied = state[q0].copy()
    assert copied.qubits == (q0, q1)