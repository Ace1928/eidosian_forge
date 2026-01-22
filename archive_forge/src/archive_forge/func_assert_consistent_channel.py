from typing import Any
import numpy as np
import cirq
def assert_consistent_channel(gate: Any, rtol: float=1e-05, atol: float=1e-08):
    """Asserts that a given gate has Kraus operators and that they are properly normalized."""
    assert cirq.has_kraus(gate), f'Given gate {gate!r} does not return True for cirq.has_kraus.'
    kraus_ops = cirq.kraus(gate)
    assert cirq.is_cptp(kraus_ops=kraus_ops, rtol=rtol, atol=atol), f'Kraus operators for {gate!r} did not sum to identity up to expected tolerances. Summed to {sum((m.T.conj() @ m for m in kraus_ops))}'