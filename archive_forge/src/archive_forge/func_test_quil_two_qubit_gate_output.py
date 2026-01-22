import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
def test_quil_two_qubit_gate_output():
    q0, q1 = _make_qubits(2)
    gate = cirq_rigetti.quil_output.QuilTwoQubitGate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    output = cirq_rigetti.quil_output.QuilOutput((gate.on(q0, q1),), (q0, q1))
    assert str(output) == '# Created using Cirq.\n\nDEFGATE USERGATE1:\n    1.0+0.0i, 0.0+0.0i, 0.0+0.0i, 0.0+0.0i\n    0.0+0.0i, 1.0+0.0i, 0.0+0.0i, 0.0+0.0i\n    0.0+0.0i, 0.0+0.0i, 1.0+0.0i, 0.0+0.0i\n    0.0+0.0i, 0.0+0.0i, 0.0+0.0i, 1.0+0.0i\nUSERGATE1 0 1\n'