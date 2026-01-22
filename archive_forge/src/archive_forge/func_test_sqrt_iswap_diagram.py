import cirq
import cirq.contrib.qcircuit as ccq
import cirq.testing as ct
def test_sqrt_iswap_diagram():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.ISWAP(q0, q1) ** 0.5)
    expected_diagram = '\n\\Qcircuit @R=1em @C=0.75em {\n \\\\\n &\\lstick{\\text{q(0)}}& \\qw&\\multigate{1}{\\text{ISWAP}^{0.5}} \\qw&\\qw\\\\\n &\\lstick{\\text{q(1)}}& \\qw&\\ghost{\\text{ISWAP}^{0.5}}        \\qw&\\qw\\\\\n \\\\\n}'.strip()
    assert_has_qcircuit_diagram(circuit, expected_diagram)