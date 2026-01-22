from typing import List
import cirq
def test_merge_single_qubit_gates_to_phxz():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(a), cirq.Y(b) ** 0.5, cirq.CZ(a, b), cirq.H(a), cirq.Z(a), cirq.measure(b, key='m'), cirq.H(a).with_classical_controls('m'))
    assert_optimizes(optimized=cirq.merge_single_qubit_gates_to_phxz(c), expected=cirq.Circuit(_phxz(-1, 1, 0).on(a), _phxz(0.5, 0.5, 0).on(b), cirq.CZ(a, b), _phxz(-0.5, 0.5, 0).on(a), cirq.measure(b, key='m'), cirq.H(a).with_classical_controls('m')))