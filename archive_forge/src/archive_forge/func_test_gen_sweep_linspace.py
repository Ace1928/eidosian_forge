import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
def test_gen_sweep_linspace():
    sweep = params_pb2.SingleSweep(parameter_key='foo', linspace=params_pb2.Linspace(first_point=0, last_point=10, num_points=11))
    out = params._sweep_from_single_param_sweep_proto(sweep)
    assert out == cirq.Linspace('foo', 0, 10, 11)