import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_measurement_gate_target_order():
    q0, _, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(q2, q0, key='tomyheart'))
    serializer = ionq.Serializer()
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(body={'gateset': 'native', 'qubits': 3, 'circuit': []}, metadata={'measurement0': f'tomyheart{chr(31)}2,0'}, settings={})