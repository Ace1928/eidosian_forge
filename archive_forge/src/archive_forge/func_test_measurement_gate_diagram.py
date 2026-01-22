from typing import cast
import numpy as np
import pytest
import cirq
def test_measurement_gate_diagram():
    assert cirq.circuit_diagram_info(cirq.MeasurementGate(1, key='test')) == cirq.CircuitDiagramInfo(("M('test')",))
    assert cirq.circuit_diagram_info(cirq.MeasurementGate(3, 'a'), cirq.CircuitDiagramInfoArgs(known_qubits=None, known_qubit_count=3, use_unicode_characters=True, precision=None, label_map=None)) == cirq.CircuitDiagramInfo(("M('a')", 'M', 'M'))
    assert cirq.circuit_diagram_info(cirq.MeasurementGate(2, 'a', invert_mask=(False, True))) == cirq.CircuitDiagramInfo(("M('a')", '!M'))
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.measure(a, b)), '\na: ───M───\n      │\nb: ───M───\n')
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.measure(a, b, invert_mask=(True,))), '\na: ───!M───\n      │\nb: ───M────\n')
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.measure(a, b, confusion_map={(1,): np.array([[0, 1], [1, 0]])})), '\na: ───M────\n      │\nb: ───?M───\n')
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.measure(a, b, invert_mask=(False, True), confusion_map={(1,): np.array([[0, 1], [1, 0]])})), '\na: ───M─────\n      │\nb: ───!?M───\n')
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.measure(a, b, key='test')), "\na: ───M('test')───\n      │\nb: ───M───────────\n")