import pytest
import networkx as nx
from networkx.utils import pairwise
def test_all_pairs_dijkstra(self):
    cycle = nx.cycle_graph(7)
    out = dict(nx.all_pairs_dijkstra(cycle))
    assert out[0][0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
    assert out[0][1][3] == [0, 1, 2, 3]
    cycle[1][2]['weight'] = 10
    out = dict(nx.all_pairs_dijkstra(cycle))
    assert out[0][0] == {0: 0, 1: 1, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
    assert out[0][1][3] == [0, 6, 5, 4, 3]