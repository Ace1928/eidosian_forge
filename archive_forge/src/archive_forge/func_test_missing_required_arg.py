import numpy as np
import pytest
import sympy
from google.protobuf import json_format
import cirq_google
from cirq_google.serialization.arg_func_langs import (
from cirq_google.api import v2
def test_missing_required_arg():
    with pytest.raises(ValueError, match='blah is missing'):
        _ = float_arg_from_proto(v2.program_pb2.FloatArg(), arg_function_language='exp', required_arg_name='blah')
    with pytest.raises(ValueError, match='unrecognized argument type'):
        _ = arg_from_proto(v2.program_pb2.Arg(), arg_function_language='exp', required_arg_name='blah')
    with pytest.raises(ValueError, match='Unrecognized function type '):
        _ = arg_from_proto(v2.program_pb2.Arg(func=v2.program_pb2.ArgFunction(type='magic')), arg_function_language='exp', required_arg_name='blah')
    assert arg_from_proto(v2.program_pb2.Arg(), arg_function_language='exp') is None