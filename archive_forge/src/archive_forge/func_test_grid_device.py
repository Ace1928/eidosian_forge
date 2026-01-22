import pytest
import networkx as nx
import cirq
def test_grid_device():
    rect_device = cirq.testing.construct_grid_device(5, 7)
    rect_device_graph = rect_device.metadata.nx_graph
    isomorphism_class = nx.Graph()
    row_edges = [(cirq.GridQubit(i, j), cirq.GridQubit(i, j + 1)) for i in range(5) for j in range(6)]
    col_edges = [(cirq.GridQubit(i, j), cirq.GridQubit(i + 1, j)) for j in range(7) for i in range(4)]
    isomorphism_class.add_edges_from(row_edges)
    isomorphism_class.add_edges_from(col_edges)
    assert all((q in rect_device_graph.nodes for q in cirq.GridQubit.rect(5, 7)))
    assert nx.is_isomorphic(isomorphism_class, rect_device_graph)