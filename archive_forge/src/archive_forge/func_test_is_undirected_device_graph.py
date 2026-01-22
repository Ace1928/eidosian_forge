import pytest
import cirq
import cirq.contrib.graph_device as ccgd
import cirq.contrib.graph_device.graph_device as ccgdgd
def test_is_undirected_device_graph():
    assert not ccgd.is_undirected_device_graph('abc')
    graph = ccgd.UndirectedHypergraph()
    assert ccgd.is_undirected_device_graph(graph)
    a, b, c, d, e = cirq.LineQubit.range(5)
    graph.add_edge((a, b))
    assert ccgd.is_undirected_device_graph(graph)
    graph.add_edge((b, c), ccgd.UnconstrainedUndirectedGraphDeviceEdge)
    assert ccgd.is_undirected_device_graph(graph)
    graph.add_edge((d, e), 'abc')
    assert not ccgd.is_undirected_device_graph(graph)
    graph = ccgd.UndirectedHypergraph(vertices=(0, 1))
    assert not ccgd.is_undirected_device_graph(graph)