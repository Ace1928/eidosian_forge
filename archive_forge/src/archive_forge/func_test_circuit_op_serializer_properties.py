import pytest
import sympy
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_circuit_op_serializer_properties():
    serializer = cg.CircuitOpSerializer()
    assert serializer.internal_type == cirq.FrozenCircuit
    assert serializer.serialized_id == 'circuit'