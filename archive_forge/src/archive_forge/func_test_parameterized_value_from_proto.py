import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_parameterized_value_from_proto():
    from_proto = programs._parameterized_value_from_proto
    m1 = operations_pb2.ParameterizedFloat(raw=5)
    assert from_proto(m1) == 5
    with pytest.raises(ValueError):
        from_proto(operations_pb2.ParameterizedFloat())
    m3 = operations_pb2.ParameterizedFloat(parameter_key='rr')
    assert from_proto(m3) == sympy.Symbol('rr')