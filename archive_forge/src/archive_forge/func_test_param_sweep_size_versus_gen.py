import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
@pytest.mark.parametrize('param_sweep', example_sweeps())
def test_param_sweep_size_versus_gen(param_sweep):
    sweep = params.sweep_from_proto(param_sweep)
    predicted_size = len(sweep)
    out = list(sweep)
    assert len(out) == predicted_size