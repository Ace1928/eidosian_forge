import pytest
import networkx as nx
from networkx.utils import pairwise
def test_negative_cycle_heuristic(self):
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=-1)
    G.add_edge(1, 2, weight=-1)
    G.add_edge(2, 3, weight=-1)
    G.add_edge(3, 0, weight=3)
    assert not nx.negative_edge_cycle(G, heuristic=True)
    G.add_edge(2, 0, weight=1.999)
    assert nx.negative_edge_cycle(G, heuristic=True)
    G.edges[2, 0]['weight'] = 2
    assert not nx.negative_edge_cycle(G, heuristic=True)