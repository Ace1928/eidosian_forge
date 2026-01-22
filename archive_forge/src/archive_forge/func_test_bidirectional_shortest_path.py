import pytest
import networkx as nx
def test_bidirectional_shortest_path(self):
    assert nx.bidirectional_shortest_path(self.cycle, 0, 3) == [0, 1, 2, 3]
    assert nx.bidirectional_shortest_path(self.cycle, 0, 4) == [0, 6, 5, 4]
    validate_grid_path(4, 4, 1, 12, nx.bidirectional_shortest_path(self.grid, 1, 12))
    assert nx.bidirectional_shortest_path(self.directed_cycle, 0, 3) == [0, 1, 2, 3]
    assert nx.bidirectional_shortest_path(self.cycle, 3, 3) == [3]