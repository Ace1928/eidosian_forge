import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_parameterized_invalid():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('x'))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='parameterized'):
        _ = serializer.serialize(circuit)