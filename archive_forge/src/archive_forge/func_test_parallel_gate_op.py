import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('gate, num_copies', [(cirq.X, 1), (cirq.Y, 2), (cirq.Z, 3), (cirq.H, 4)])
def test_parallel_gate_op(gate, num_copies):
    qubits = cirq.LineQubit.range(num_copies * gate.num_qubits())
    assert cirq.parallel_gate_op(gate, *qubits) == cirq.ParallelGate(gate, num_copies).on(*qubits)