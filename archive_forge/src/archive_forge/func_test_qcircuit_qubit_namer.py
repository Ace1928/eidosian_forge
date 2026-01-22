import cirq
import cirq.contrib.qcircuit as ccq
import cirq.testing as ct
def test_qcircuit_qubit_namer():
    from cirq.contrib.qcircuit import qcircuit_diagram
    assert qcircuit_diagram.qcircuit_qubit_namer(cirq.NamedQubit('q')) == '\\lstick{\\text{q}}&'
    assert qcircuit_diagram.qcircuit_qubit_namer(cirq.NamedQubit('q_1')) == '\\lstick{\\text{q\\_1}}&'
    assert qcircuit_diagram.qcircuit_qubit_namer(cirq.NamedQubit('q^1')) == '\\lstick{\\text{q\\textasciicircum{}1}}&'
    assert qcircuit_diagram.qcircuit_qubit_namer(cirq.NamedQubit('q_{1}')) == '\\lstick{\\text{q\\_\\{1\\}}}&'