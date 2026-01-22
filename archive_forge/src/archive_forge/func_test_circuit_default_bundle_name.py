import cirq
import cirq_web
import pytest
def test_circuit_default_bundle_name():
    qubits = [cirq.GridQubit(x, y) for x in range(2) for y in range(2)]
    moment = cirq.Moment(cirq.H(qubits[0]))
    circuit = cirq_web.Circuit3D(cirq.Circuit(moment))
    assert circuit.get_widget_bundle_name() == 'circuit.bundle.js'