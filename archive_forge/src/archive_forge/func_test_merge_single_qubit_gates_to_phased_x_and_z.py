from typing import List
import cirq
def test_merge_single_qubit_gates_to_phased_x_and_z():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(a), cirq.Y(b) ** 0.5, cirq.CZ(a, b), cirq.H(a), cirq.Z(a), cirq.measure(b, key='m'), cirq.H(a).with_classical_controls('m'))
    assert_optimizes(optimized=cirq.merge_single_qubit_gates_to_phased_x_and_z(c), expected=cirq.Circuit(cirq.PhasedXPowGate(phase_exponent=1)(a), cirq.Y(b) ** 0.5, cirq.CZ(a, b), cirq.PhasedXPowGate(phase_exponent=-0.5)(a) ** 0.5, cirq.measure(b, key='m'), cirq.H(a).with_classical_controls('m')))