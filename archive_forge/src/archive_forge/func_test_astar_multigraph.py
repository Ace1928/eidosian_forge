import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_multigraph(self):
    G = nx.MultiDiGraph(self.XG)
    G.add_weighted_edges_from(((u, v, 1000) for u, v in list(G.edges())))
    assert nx.astar_path(G, 's', 'v') == ['s', 'x', 'u', 'v']
    assert nx.astar_path_length(G, 's', 'v') == 9