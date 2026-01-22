import pytest
import networkx as nx
from networkx.utils import pairwise
def test_dijkstra_pred_distance_multigraph(self):
    G = nx.MultiGraph()
    G.add_edge('a', 'b', key='short', foo=5, weight=100)
    G.add_edge('a', 'b', key='long', bar=1, weight=110)
    p, d = nx.dijkstra_predecessor_and_distance(G, 'a')
    assert p == {'a': [], 'b': ['a']}
    assert d == {'a': 0, 'b': 100}