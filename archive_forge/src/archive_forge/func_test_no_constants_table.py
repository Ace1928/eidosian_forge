from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_no_constants_table():
    serializer = cg.CircuitSerializer()
    op = op_proto({'xpowgate': {'exponent': {'float_value': 1.0}}, 'qubits': [{'id': '1_2'}], 'token_constant_index': 0})
    with pytest.raises(ValueError, match='Proto has references to constants table'):
        serializer._deserialize_gate_op(op)