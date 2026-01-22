import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_inadmissible_heuristic_with_cutoff(self):
    heuristic_values = {'s': 36, 'y': 14, 'x': 10, 'u': 10, 'v': 0}

    def h(u, v):
        return heuristic_values[u]
    assert nx.astar_path_length(self.XG, 's', 'v', heuristic=h) == 10
    assert nx.astar_path_length(self.XG, 's', 'v', heuristic=h, cutoff=15) == 10
    with pytest.raises(nx.NetworkXNoPath):
        nx.astar_path_length(self.XG, 's', 'v', heuristic=h, cutoff=9)
    with pytest.raises(nx.NetworkXNoPath):
        nx.astar_path_length(self.XG, 's', 'v', heuristic=h, cutoff=12)