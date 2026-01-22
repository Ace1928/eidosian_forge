import pytest
import networkx as nx
@pytest.mark.parametrize('graph_type', [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_summarization_empty(graph_type):
    G = graph_type()
    summary_graph = nx.snap_aggregation(G, node_attributes=('color',))
    assert nx.is_isomorphic(summary_graph, G)