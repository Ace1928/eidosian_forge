import pytest
import sympy
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_circuit_op_from_proto_errors():
    deserializer = cg.CircuitOpDeserializer()
    serialized = v2.program_pb2.CircuitOperation(circuit_constant_index=1)
    constants = [v2.program_pb2.Constant(string_value=DEFAULT_TOKEN), v2.program_pb2.Constant(circuit_value=default_circuit_proto())]
    bad_deserialized_constants = [DEFAULT_TOKEN]
    with pytest.raises(ValueError, match='does not appear in the deserialized_constants list'):
        deserializer.from_proto(serialized, constants=constants, deserialized_constants=bad_deserialized_constants)
    bad_deserialized_constants = [DEFAULT_TOKEN, 2]
    with pytest.raises(ValueError, match='Constant at index 1 was expected to be a circuit'):
        deserializer.from_proto(serialized, constants=constants, deserialized_constants=bad_deserialized_constants)