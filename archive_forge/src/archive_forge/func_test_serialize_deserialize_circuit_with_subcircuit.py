from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_serialize_deserialize_circuit_with_subcircuit():
    serializer = cg.CircuitSerializer()
    tag1 = cg.CalibrationTag('abc123')
    fcircuit = cirq.FrozenCircuit(cirq.XPowGate(exponent=2 * sympy.Symbol('t'))(Q0))
    circuit = cirq.Circuit(cirq.X(Q1).with_tags(tag1), cirq.CircuitOperation(fcircuit).repeat(repetition_ids=['a', 'b']), cirq.CircuitOperation(fcircuit).with_qubit_mapping({Q0: Q1}), cirq.X(Q0))
    op_x = v2.program_pb2.Operation()
    op_x.xpowgate.exponent.float_value = 1.0
    op_x.qubit_constant_index.append(2)
    op_tag = v2.program_pb2.Operation()
    op_tag.xpowgate.exponent.float_value = 1.0
    op_tag.qubit_constant_index.append(0)
    op_tag.token_constant_index = 1
    op_symbol = v2.program_pb2.Operation()
    op_symbol.xpowgate.exponent.func.type = 'mul'
    op_symbol.xpowgate.exponent.func.args.add().arg_value.float_value = 2.0
    op_symbol.xpowgate.exponent.func.args.add().symbol = 't'
    op_symbol.qubit_constant_index.append(2)
    c_op1 = v2.program_pb2.CircuitOperation()
    c_op1.circuit_constant_index = 3
    rep_spec = c_op1.repetition_specification
    rep_spec.repetition_count = 2
    rep_spec.repetition_ids.ids.extend(['a', 'b'])
    c_op2 = v2.program_pb2.CircuitOperation()
    c_op2.circuit_constant_index = 3
    c_op2.repetition_specification.repetition_count = 1
    qmap = c_op2.qubit_map.entries.add()
    qmap.key.id = '2_4'
    qmap.value.id = '2_5'
    proto = v2.program_pb2.Program(language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME), circuit=v2.program_pb2.Circuit(scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[v2.program_pb2.Moment(operations=[op_tag], circuit_operations=[c_op1]), v2.program_pb2.Moment(operations=[op_x], circuit_operations=[c_op2])]), constants=[v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='2_5')), v2.program_pb2.Constant(string_value='abc123'), v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='2_4')), v2.program_pb2.Constant(circuit_value=v2.program_pb2.Circuit(scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[v2.program_pb2.Moment(operations=[op_symbol])]))])
    assert proto == serializer.serialize(circuit)
    assert serializer.deserialize(proto) == circuit