import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
@pytest.mark.parametrize('gate', [cirq.MatrixGate(cirq.unitary(cirq.CX), qid_shape=(2, 2)), cirq.ISWAP, cirq.SWAP, cirq.CNOT, cirq.CZ, cirq.PhasedISwapPowGate(exponent=1.0), cirq.PhasedISwapPowGate(exponent=1.0, phase_exponent=0.33), cirq.PhasedISwapPowGate(exponent=0.66, phase_exponent=0.25), *[cirq.givens(theta) for theta in np.linspace(0, 2 * np.pi, 30)], *[cirq.ZZPowGate(exponent=2 * phi / np.pi) for phi in np.linspace(0, 2 * np.pi, 30)], *[cirq.CZPowGate(exponent=phi / np.pi) for phi in np.linspace(0, 2 * np.pi, 30)]])
def test_convert_to_sycamore_equivalent_unitaries(gate):
    circuit = cirq.Circuit(gate.on(*cirq.LineQubit.range(2)))
    converted_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset())
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, converted_circuit, atol=1e-08)