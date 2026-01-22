from functools import partial
import pytest
import networkx as nx
def test_descendants_at_distance_missing_source(self):
    with pytest.raises(nx.NetworkXError):
        nx.descendants_at_distance(self.G, 'abc', 0)