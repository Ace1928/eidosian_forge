import cirq
import cirq.contrib.qcircuit as ccq
import cirq.testing as ct
def test_other_diagram():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.X(a), cirq.Y(b), cirq.Z(c))
    expected_diagram = '\n\\Qcircuit @R=1em @C=0.75em {\n \\\\\n &\\lstick{\\text{q(0)}}& \\qw&\\targ           \\qw&\\qw\\\\\n &\\lstick{\\text{q(1)}}& \\qw&\\gate{\\text{Y}} \\qw&\\qw\\\\\n &\\lstick{\\text{q(2)}}& \\qw&\\gate{\\text{Z}} \\qw&\\qw\\\\\n \\\\\n}'.strip()
    assert_has_qcircuit_diagram(circuit, expected_diagram)