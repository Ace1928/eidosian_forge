import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_atol():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer(atol=0.1)
    circuit = cirq.Circuit(cirq.X(q0) ** 1.09)
    result = serializer.serialize(circuit)
    assert result.body['circuit'][0]['gate'] == 'x'