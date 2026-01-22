import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
def test_i_swap_with_power():
    q0, q1 = _make_qubits(2)
    output = QuilOutput((cirq.ISWAP(q0, q1) ** 0.25,), (q0, q1))
    assert str(output) == f'# Created using Cirq.\n\nXY({np.pi / 4}) 0 1\n'