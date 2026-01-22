import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_admissible_heuristic_with_cutoff(self):
    heuristic_values = {'s': 36, 'y': 4, 'x': 0, 'u': 0, 'v': 0}

    def h(u, v):
        return heuristic_values[u]
    assert nx.astar_path_length(self.XG, 's', 'v') == 9
    assert nx.astar_path_length(self.XG, 's', 'v', heuristic=h) == 9
    assert nx.astar_path_length(self.XG, 's', 'v', heuristic=h, cutoff=12) == 9
    assert nx.astar_path_length(self.XG, 's', 'v', heuristic=h, cutoff=9) == 9
    with pytest.raises(nx.NetworkXNoPath):
        nx.astar_path_length(self.XG, 's', 'v', heuristic=h, cutoff=8)