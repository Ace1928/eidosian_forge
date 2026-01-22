import pytest
import networkx as nx
from networkx.generators import directed
def test_empty_graph_hash():
    """
    empty graphs should give hashes regardless of other params
    """
    G1 = nx.empty_graph()
    G2 = nx.empty_graph()
    h1 = nx.weisfeiler_lehman_graph_hash(G1)
    h2 = nx.weisfeiler_lehman_graph_hash(G2)
    h3 = nx.weisfeiler_lehman_graph_hash(G2, edge_attr='edge_attr1')
    h4 = nx.weisfeiler_lehman_graph_hash(G2, node_attr='node_attr1')
    h5 = nx.weisfeiler_lehman_graph_hash(G2, edge_attr='edge_attr1', node_attr='node_attr1')
    h6 = nx.weisfeiler_lehman_graph_hash(G2, iterations=10)
    assert h1 == h2
    assert h1 == h3
    assert h1 == h4
    assert h1 == h5
    assert h1 == h6