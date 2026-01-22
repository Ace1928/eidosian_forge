import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def assert_proto_dict_convert(gate: cirq.Gate, proto: operations_pb2.Operation, *qubits: cirq.Qid):
    assert programs.gate_to_proto(gate, qubits, delay=0) == proto
    assert programs.xmon_op_from_proto(proto) == gate(*qubits)