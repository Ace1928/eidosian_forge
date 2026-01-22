import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_project_weighted_ratio(self):
    edges = [('A', 'B', 2 / 6.0), ('A', 'C', 1 / 6.0), ('B', 'C', 1 / 6.0), ('B', 'D', 1 / 6.0), ('B', 'E', 2 / 6.0), ('E', 'F', 1 / 6.0)]
    Panswer = nx.Graph()
    Panswer.add_weighted_edges_from(edges)
    P = bipartite.weighted_projected_graph(self.G, 'ABCDEF', ratio=True)
    assert edges_equal(list(P.edges()), Panswer.edges())
    for u, v in list(P.edges()):
        assert P[u][v]['weight'] == Panswer[u][v]['weight']
    edges = [('A', 'B', 3 / 3.0), ('A', 'E', 1 / 3.0), ('A', 'C', 1 / 3.0), ('A', 'D', 1 / 3.0), ('B', 'E', 1 / 3.0), ('B', 'C', 1 / 3.0), ('B', 'D', 1 / 3.0), ('C', 'D', 1 / 3.0)]
    Panswer = nx.Graph()
    Panswer.add_weighted_edges_from(edges)
    P = bipartite.weighted_projected_graph(self.N, 'ABCDE', ratio=True)
    assert edges_equal(list(P.edges()), Panswer.edges())
    for u, v in list(P.edges()):
        assert P[u][v]['weight'] == Panswer[u][v]['weight']