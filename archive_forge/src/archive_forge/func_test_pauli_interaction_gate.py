import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
def test_pauli_interaction_gate():
    q0, q1 = _make_qubits(2)
    output = cirq_rigetti.quil_output.QuilOutput(PauliInteractionGate.CZ.on(q0, q1), (q0, q1))
    assert str(output) == '# Created using Cirq.\n\nCZ 0 1\n'