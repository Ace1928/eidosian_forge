from functools import partial
import pytest
import networkx as nx
def test_limited_descendants_at_distance(self):
    for distance, descendants in enumerate([{0}, {1}, {2}, {3, 7}, {4, 8}, {5, 9}, {6, 10}]):
        assert nx.descendants_at_distance(self.G, 0, distance) == descendants
    for distance, descendants in enumerate([{2}, {3, 7}, {8}, {9}, {10}]):
        assert nx.descendants_at_distance(self.D, 2, distance) == descendants