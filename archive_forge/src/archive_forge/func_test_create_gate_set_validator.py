import pytest
import cirq
import cirq_google as cg
import cirq_google.engine.engine_validator as engine_validator
def test_create_gate_set_validator():
    circuit = _big_circuit(4)
    smaller_size_validator = engine_validator.create_program_validator(max_size=10000)
    smaller_size_validator([circuit] * 2, [{}] * 2, 1000, cg.CIRCUIT_SERIALIZER)
    with pytest.raises(RuntimeError, match='Program too long'):
        smaller_size_validator([circuit] * 5, [{}] * 5, 1000, cg.CIRCUIT_SERIALIZER)
    larger_size_validator = engine_validator.create_program_validator(max_size=500000)
    larger_size_validator([circuit] * 10, [{}] * 10, 1000, cg.CIRCUIT_SERIALIZER)