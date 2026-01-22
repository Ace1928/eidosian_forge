import cirq
import pytest
def test_directed_device():
    device = cirq.testing.construct_ring_device(10, directed=True)
    device_graph = device.metadata.nx_graph
    with pytest.raises(ValueError, match='Device graph must be undirected.'):
        cirq.RouteCQC(device_graph)