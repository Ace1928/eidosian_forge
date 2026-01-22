import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
def test_unconveritble_op():
    q0, = _make_qubits(1)

    class MyGate(cirq.Gate):

        def num_qubits(self) -> int:
            return 1
    op = MyGate()(q0)
    with pytest.raises(ValueError, match="Can't convert"):
        _ = cirq_rigetti.quil_output.QuilOutput(op, (q0,))._op_to_quil(op)