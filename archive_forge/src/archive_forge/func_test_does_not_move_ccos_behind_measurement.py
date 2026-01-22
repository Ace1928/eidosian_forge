import pytest
import cirq
def test_does_not_move_ccos_behind_measurement():
    q = cirq.LineQubit.range(3)
    c_orig = cirq.Circuit(cirq.measure(q[0], key='m'), cirq.X(q[1]).with_classical_controls('m'), cirq.Moment(cirq.X.on_each(q[1], q[2])))
    cirq.testing.assert_has_diagram(c_orig, '\n0: ───M───────────\n      ║\n1: ───╫───X───X───\n      ║   ║\n2: ───╫───╫───X───\n      ║   ║\nm: ═══@═══^═══════\n')
    c_out = cirq.stratified_circuit(c_orig, categories=[cirq.GateOperation, cirq.ClassicallyControlledOperation])
    cirq.testing.assert_has_diagram(c_out, '\n      ┌──┐\n0: ────M─────────────\n       ║\n1: ────╫─────X───X───\n       ║     ║\n2: ────╫X────╫───────\n       ║     ║\nm: ════@═════^═══════\n      └──┘\n')