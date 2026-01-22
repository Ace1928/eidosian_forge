import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_w1(self):
    G = nx.DiGraph()
    G.add_edges_from([('s', 'u'), ('s', 'x'), ('u', 'v'), ('u', 'x'), ('v', 'y'), ('x', 'u'), ('x', 'w'), ('w', 'v'), ('x', 'y'), ('y', 's'), ('y', 'v')])
    assert nx.astar_path(G, 's', 'v') == ['s', 'u', 'v']
    assert nx.astar_path_length(G, 's', 'v') == 2