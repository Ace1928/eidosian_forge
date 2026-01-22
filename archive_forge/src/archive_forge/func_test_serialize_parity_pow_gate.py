import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_parity_pow_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    for gate, name in ((cirq.XXPowGate, 'xx'), (cirq.YYPowGate, 'yy'), (cirq.ZZPowGate, 'zz')):
        for exponent in (0.5, 1.0, 1.5):
            circuit = cirq.Circuit(gate(exponent=exponent)(q0, q1))
            result = serializer.serialize(circuit)
            assert result == ionq.SerializedProgram(body={'gateset': 'qis', 'qubits': 2, 'circuit': [{'gate': name, 'targets': [0, 1], 'rotation': exponent * np.pi}]}, metadata={}, settings={})