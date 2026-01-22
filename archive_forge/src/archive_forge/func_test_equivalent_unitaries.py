import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
def test_equivalent_unitaries():
    """This test covers the factor of pi change. However, it will be skipped
    if pyquil is unavailable for import.

    References:
        https://docs.pytest.org/en/latest/skipping.html#skipping-on-a-missing-import-dependency
    """
    pyquil = pytest.importorskip('pyquil')
    pyquil_simulation_tools = pytest.importorskip('pyquil.simulation.tools')
    q0, q1 = _make_qubits(2)
    operations = [cirq.XPowGate(exponent=0.5, global_shift=-0.5)(q0), cirq.YPowGate(exponent=0.5, global_shift=-0.5)(q0), cirq.ZPowGate(exponent=0.5, global_shift=-0.5)(q0), cirq.CZPowGate(exponent=0.5)(q0, q1), cirq.ISwapPowGate(exponent=0.5)(q0, q1)]
    output = cirq_rigetti.quil_output.QuilOutput(operations, (q0, q1))
    program = pyquil.Program(str(output))
    pyquil_unitary = pyquil_simulation_tools.program_unitary(program, n_qubits=2)
    cirq_unitary = cirq.Circuit(cirq.SWAP(q0, q1), operations, cirq.SWAP(q0, q1)).unitary()
    assert np.allclose(pyquil_unitary, cirq_unitary)