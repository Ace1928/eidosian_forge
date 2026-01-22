import math
import cirq
import numpy
import pytest
import cirq_ionq as ionq
@pytest.mark.parametrize('gate,nqubits,diagram', [(ionq.GPIGate(phi=0.1), 1, '0: ───GPI(0.1)───'), (ionq.GPI2Gate(phi=0.2), 1, '0: ───GPI2(0.2)───'), (ionq.MSGate(phi0=0.1, phi1=0.2), 2, '0: ───MS(0.1)───\n      │\n1: ───MS(0.2)───')])
def test_gate_methods(gate, nqubits, diagram):
    assert str(gate) != ''
    assert repr(gate) != ''
    assert gate.num_qubits() == nqubits
    assert cirq.protocols.circuit_diagram_info(gate) is not None
    c = cirq.Circuit()
    c.append([gate.on(*cirq.LineQubit.range(nqubits))])
    assert c.to_text_diagram() == diagram