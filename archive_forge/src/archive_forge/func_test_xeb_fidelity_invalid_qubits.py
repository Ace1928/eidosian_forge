import itertools
from typing import Sequence
import numpy as np
import pytest
import cirq
def test_xeb_fidelity_invalid_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    bitstrings = sample_noisy_bitstrings(circuit, (q0, q1, q2), 0.9, 10)
    with pytest.raises(ValueError):
        cirq.xeb_fidelity(circuit, bitstrings, (q0, q2))