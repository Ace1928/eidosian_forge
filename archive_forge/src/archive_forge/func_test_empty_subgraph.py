import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_empty_subgraph(self):
    nullgraph = nx.null_graph()
    E5 = nx.empty_graph(5)
    E10 = nx.empty_graph(10)
    H = E10.subgraph([])
    assert nx.is_isomorphic(H, nullgraph)
    H = E10.subgraph([1, 2, 3, 4, 5])
    assert nx.is_isomorphic(H, E5)