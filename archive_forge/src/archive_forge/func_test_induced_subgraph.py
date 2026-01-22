from networkx.utils.misc import graphs_equal
import pytest
import networkx as nx
import cirq
def test_induced_subgraph():
    device_graph, initial_mapping, _ = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)
    expected_induced_subgraph = nx.Graph([(cirq.NamedQubit('a'), cirq.NamedQubit('b')), (cirq.NamedQubit('b'), cirq.NamedQubit('c')), (cirq.NamedQubit('c'), cirq.NamedQubit('d'))])
    assert graphs_equal(mm.induced_subgraph_int, nx.relabel_nodes(expected_induced_subgraph, mm.physical_qid_to_int))