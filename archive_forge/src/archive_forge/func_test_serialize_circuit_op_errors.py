from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_serialize_circuit_op_errors():
    serializer = cg.CircuitSerializer()
    constants = [default_circuit_proto()]
    raw_constants = {default_circuit(): 0}
    op = cirq.CircuitOperation(default_circuit())
    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        serializer._serialize_circuit_op(op)
    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        serializer._serialize_circuit_op(op, constants=constants)
    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        serializer._serialize_circuit_op(op, raw_constants=raw_constants)