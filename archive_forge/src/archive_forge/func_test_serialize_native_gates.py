import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_native_gates():
    q0, q1, q2 = cirq.LineQubit.range(3)
    gpi = ionq.GPIGate(phi=0.1).on(q0)
    gpi2 = ionq.GPI2Gate(phi=0.2).on(q1)
    ms = ionq.MSGate(phi0=0.3, phi1=0.4).on(q1, q2)
    circuit = cirq.Circuit([gpi, gpi2, ms])
    serializer = ionq.Serializer()
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(body={'gateset': 'native', 'qubits': 3, 'circuit': [{'gate': 'gpi', 'target': 0, 'phase': 0.1}, {'gate': 'gpi2', 'target': 1, 'phase': 0.2}, {'gate': 'ms', 'targets': [1, 2], 'phases': [0.3, 0.4], 'angle': 0.25}]}, metadata={}, settings={})