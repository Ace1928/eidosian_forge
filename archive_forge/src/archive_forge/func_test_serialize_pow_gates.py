import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_pow_gates():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    for name, gate in (('rx', cirq.X), ('ry', cirq.Y), ('rz', cirq.Z)):
        for exponent in (1.1, 0.6):
            circuit = cirq.Circuit((gate ** exponent)(q0))
            result = serializer.serialize(circuit)
            assert result == ionq.SerializedProgram(body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': name, 'targets': [0], 'rotation': exponent * np.pi}]}, metadata={}, settings={})