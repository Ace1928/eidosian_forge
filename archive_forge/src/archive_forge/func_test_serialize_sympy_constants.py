import numpy as np
import pytest
import sympy
from google.protobuf import json_format
import cirq_google
from cirq_google.serialization.arg_func_langs import (
from cirq_google.api import v2
def test_serialize_sympy_constants():
    proto = arg_to_proto(sympy.pi, arg_function_language='')
    packed = json_format.MessageToDict(proto, including_default_value_fields=True, preserving_proto_field_name=True, use_integers_for_enums=True)
    assert len(packed) == 1
    assert len(packed['arg_value']) == 1
    assert np.isclose(packed['arg_value']['float_value'], np.float32(sympy.pi), atol=1e-07)