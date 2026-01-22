import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_settings():
    q0 = cirq.LineQubit(2)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    result = serializer.serialize(circuit, job_settings={'foo': 'bar', 'key': 'heart'})
    assert result == ionq.SerializedProgram(body={'gateset': 'qis', 'qubits': 3, 'circuit': [{'gate': 'x', 'targets': [2]}]}, metadata={}, settings={'foo': 'bar', 'key': 'heart'})