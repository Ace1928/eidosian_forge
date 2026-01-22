import pytest
import cirq
import cirq.contrib.graph_device as ccgd
def test_negative_arity_arg_uniform_undirected_linear_device():
    with pytest.raises(ValueError):
        ccgd.uniform_undirected_linear_device(5, {-1: None})
    with pytest.raises(ValueError):
        ccgd.uniform_undirected_linear_device(5, {0: None})