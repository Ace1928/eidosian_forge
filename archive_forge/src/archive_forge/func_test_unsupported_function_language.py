import numpy as np
import pytest
import sympy
from google.protobuf import json_format
import cirq_google
from cirq_google.serialization.arg_func_langs import (
from cirq_google.api import v2
def test_unsupported_function_language():
    with pytest.raises(ValueError, match='Unrecognized arg_function_language'):
        _ = arg_to_proto(sympy.Symbol('a') + sympy.Symbol('b'), arg_function_language='NEVER GONNAH APPEN')
    with pytest.raises(ValueError, match='Unrecognized arg_function_language'):
        _ = arg_to_proto(3 * sympy.Symbol('b'), arg_function_language='NEVER GONNAH APPEN')
    with pytest.raises(ValueError, match='Unrecognized arg_function_language'):
        _ = arg_from_proto(v2.program_pb2.Arg(func=v2.program_pb2.ArgFunction(type='add', args=[v2.program_pb2.Arg(symbol='a'), v2.program_pb2.Arg(symbol='b')])), arg_function_language='NEVER GONNAH APPEN')