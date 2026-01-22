import numpy as np
import pytest
import sympy
from google.protobuf import json_format
import cirq_google
from cirq_google.serialization.arg_func_langs import (
from cirq_google.api import v2
@pytest.mark.parametrize('value,proto', [((True, False), {'arg_value': {'bool_values': {'values': [True, False]}}}), (np.array([True, False], dtype=bool), {'arg_value': {'bool_values': {'values': [True, False]}}})])
def test_serialize_conversion(value: ARG_LIKE, proto: v2.program_pb2.Arg):
    msg = v2.program_pb2.Arg()
    json_format.ParseDict(proto, msg)
    packed = json_format.MessageToDict(arg_to_proto(value, arg_function_language=''), including_default_value_fields=True, preserving_proto_field_name=True, use_integers_for_enums=True)
    assert packed == proto