import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
def test_sweep_from_proto_missing_type():
    ps = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(factors=[params_pb2.ZipSweep(sweeps=[params_pb2.SingleSweep(parameter_key='foo')])]))
    with pytest.raises(ValueError):
        params.sweep_from_proto(ps)