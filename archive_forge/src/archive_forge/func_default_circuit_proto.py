import pytest
import sympy
import cirq
import cirq_google as cg
from cirq_google.api import v2
def default_circuit_proto():
    op1 = v2.program_pb2.Operation()
    op1.gate.id = 'x_pow'
    op1.args['half_turns'].arg_value.string_value = 'k'
    op1.qubits.add().id = '1_1'
    op2 = v2.program_pb2.Operation()
    op2.gate.id = 'x_pow'
    op2.args['half_turns'].arg_value.float_value = 1.0
    op2.qubits.add().id = '1_2'
    op2.token_constant_index = 0
    return v2.program_pb2.Circuit(scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[v2.program_pb2.Moment(operations=[op1, op2])])