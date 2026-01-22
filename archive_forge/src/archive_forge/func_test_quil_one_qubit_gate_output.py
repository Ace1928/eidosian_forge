import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
def test_quil_one_qubit_gate_output():
    q0, = _make_qubits(1)
    gate = cirq_rigetti.quil_output.QuilOneQubitGate(np.array([[1, 0], [0, 1]]))
    output = cirq_rigetti.quil_output.QuilOutput((gate.on(q0),), (q0,))
    assert str(output) == '# Created using Cirq.\n\nDEFGATE USERGATE1:\n    1.0+0.0i, 0.0+0.0i\n    0.0+0.0i, 1.0+0.0i\nUSERGATE1 0\n'