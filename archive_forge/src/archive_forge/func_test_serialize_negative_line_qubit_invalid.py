import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_negative_line_qubit_invalid():
    q0 = cirq.LineQubit(-1)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='-1'):
        _ = serializer.serialize(circuit)