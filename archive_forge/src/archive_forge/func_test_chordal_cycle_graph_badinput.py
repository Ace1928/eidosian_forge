import pytest
import networkx as nx
@pytest.mark.parametrize('graph_type', (nx.Graph, nx.DiGraph, nx.MultiDiGraph))
def test_chordal_cycle_graph_badinput(graph_type):
    with pytest.raises(nx.NetworkXError, match='`create_using` must be an undirected multigraph'):
        nx.chordal_cycle_graph(3, create_using=graph_type)