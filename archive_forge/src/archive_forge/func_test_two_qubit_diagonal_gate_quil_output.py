import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
def test_two_qubit_diagonal_gate_quil_output():
    pyquil = pytest.importorskip('pyquil')
    pyquil_simulation_tools = pytest.importorskip('pyquil.simulation.tools')
    q0, q1 = _make_qubits(2)
    operations = [cirq.TwoQubitDiagonalGate([np.pi / 2, 0, 0, 0])(q0, q1), cirq.TwoQubitDiagonalGate([0, np.pi / 2, 0, 0])(q0, q1), cirq.TwoQubitDiagonalGate([0, 0, np.pi / 2, 0])(q0, q1), cirq.TwoQubitDiagonalGate([0, 0, 0, np.pi / 2])(q0, q1)]
    output = cirq_rigetti.quil_output.QuilOutput(operations, (q0, q1))
    program = pyquil.Program(str(output))
    assert f'\n{program.out()}' == QUIL_CPHASES_PROGRAM
    pyquil_unitary = pyquil_simulation_tools.program_unitary(program, n_qubits=2)
    cirq_unitary = cirq.Circuit(cirq.SWAP(q0, q1), operations, cirq.SWAP(q0, q1)).unitary()
    assert np.allclose(pyquil_unitary, cirq_unitary)
    operations = [cirq.TwoQubitDiagonalGate([0, 0, 0, 0])(q0, q1)]
    output = cirq_rigetti.quil_output.QuilOutput(operations, (q0, q1))
    program = pyquil.Program(str(output))
    assert f'\n{program.out()}' == QUIL_DIAGONAL_DECOMPOSE_PROGRAM