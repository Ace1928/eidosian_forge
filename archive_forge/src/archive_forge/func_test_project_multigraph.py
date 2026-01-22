import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_project_multigraph(self):
    G = nx.Graph()
    G.add_edge('a', 1)
    G.add_edge('b', 1)
    G.add_edge('a', 2)
    G.add_edge('b', 2)
    P = bipartite.projected_graph(G, 'ab')
    assert edges_equal(list(P.edges()), [('a', 'b')])
    P = bipartite.weighted_projected_graph(G, 'ab')
    assert edges_equal(list(P.edges()), [('a', 'b')])
    P = bipartite.projected_graph(G, 'ab', multigraph=True)
    assert edges_equal(list(P.edges()), [('a', 'b'), ('a', 'b')])