from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
@pytest.mark.parametrize(('op', 'op_proto'), OPERATIONS)
def test_serialize_deserialize_ops(op, op_proto):
    serializer = cg.CircuitSerializer()
    constants = []
    for q in op.qubits:
        constants.append(v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id=f'{q.row}_{q.col}')))
    circuit = cirq.Circuit(op)
    circuit_proto = v2.program_pb2.Program(language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME), circuit=v2.program_pb2.Circuit(scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[v2.program_pb2.Moment(operations=[op_proto])]), constants=constants)
    assert circuit_proto == serializer.serialize(circuit)
    assert serializer.deserialize(circuit_proto) == circuit