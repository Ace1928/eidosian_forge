from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_deserialize_circuit_with_token_strings():
    """Supporting token strings for backwards compatibility."""
    serializer = cg.CircuitSerializer()
    proto = v2.program_pb2.Program(language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME), circuit=v2.program_pb2.Circuit(scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[v2.program_pb2.Moment(operations=[v2.program_pb2.Operation(xpowgate=v2.program_pb2.XPowGate(exponent=v2.program_pb2.FloatArg(float_value=1.0)), token_value='abc123', qubit_constant_index=[0])])]), constants=[v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='2_4'))])
    tag = cg.CalibrationTag('abc123')
    circuit = cirq.Circuit(cirq.X(Q0).with_tags(tag))
    assert serializer.deserialize(proto) == circuit