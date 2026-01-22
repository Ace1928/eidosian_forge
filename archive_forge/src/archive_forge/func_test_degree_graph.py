import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_degree_graph(self):
    P3 = nx.path_graph(3)
    P5 = nx.path_graph(5)
    assert dict((d for n, d in P3.degree(['A', 'B']))) == {}
    assert sorted((d for n, d in P5.degree(P3))) == [1, 2, 2]
    assert sorted((d for n, d in P3.degree(P5))) == [1, 1, 2]
    assert list(P5.degree([])) == []
    assert dict(P5.degree([])) == {}