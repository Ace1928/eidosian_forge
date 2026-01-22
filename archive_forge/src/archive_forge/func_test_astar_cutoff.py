import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_cutoff(self):
    with pytest.raises(nx.NetworkXNoPath):
        nx.astar_path(self.XG, 's', 'v', cutoff=8.0)
    with pytest.raises(nx.NetworkXNoPath):
        nx.astar_path_length(self.XG, 's', 'v', cutoff=8.0)