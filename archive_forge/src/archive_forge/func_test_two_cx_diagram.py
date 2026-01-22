import cirq
import cirq.contrib.qcircuit as ccq
import cirq.testing as ct
def test_two_cx_diagram():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.CX(q0, q2), cirq.CX(q1, q3), cirq.CX(q0, q2), cirq.CX(q1, q3))
    expected_diagram = '\n\\Qcircuit @R=1em @C=0.75em {\n \\\\\n &\\lstick{\\text{q(0)}}& \\qw&\\control \\qw    &         \\qw    &\\control \\qw    &         \\qw    &\\qw\\\\\n &\\lstick{\\text{q(1)}}& \\qw&         \\qw\\qwx&\\control \\qw    &         \\qw\\qwx&\\control \\qw    &\\qw\\\\\n &\\lstick{\\text{q(2)}}& \\qw&\\targ    \\qw\\qwx&         \\qw\\qwx&\\targ    \\qw\\qwx&         \\qw\\qwx&\\qw\\\\\n &\\lstick{\\text{q(3)}}& \\qw&         \\qw    &\\targ    \\qw\\qwx&         \\qw    &\\targ    \\qw\\qwx&\\qw\\\\\n \\\\\n}'.strip()
    assert_has_qcircuit_diagram(circuit, expected_diagram)