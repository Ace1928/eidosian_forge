import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize('G', (nx.MultiGraph(), nx.MultiDiGraph()))
def test_restricted_view_multi(G):
    G.add_edges_from([(0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 2, 0)])
    G.add_node(4)
    H = nx.restricted_view(G, [0, 2, 5], [(1, 2, 0), (3, 4, 0)])
    assert set(H.nodes()) == {1, 3, 4}
    assert set(H.edges()) == {(1, 1)}