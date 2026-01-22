import pytest
import networkx as nx
def test_single_source_shortest_path(self):
    p = nx.shortest_path(self.cycle, 0)
    assert p[3] == [0, 1, 2, 3]
    assert p == nx.single_source_shortest_path(self.cycle, 0)
    p = nx.shortest_path(self.grid, 1)
    validate_grid_path(4, 4, 1, 12, p[12])
    p = nx.shortest_path(self.cycle, 0, weight='weight')
    assert p[3] == [0, 1, 2, 3]
    assert p == nx.single_source_dijkstra_path(self.cycle, 0)
    p = nx.shortest_path(self.grid, 1, weight='weight')
    validate_grid_path(4, 4, 1, 12, p[12])
    p = nx.shortest_path(self.cycle, 0, method='dijkstra', weight='weight')
    assert p[3] == [0, 1, 2, 3]
    assert p == nx.single_source_shortest_path(self.cycle, 0)
    p = nx.shortest_path(self.cycle, 0, method='bellman-ford', weight='weight')
    assert p[3] == [0, 1, 2, 3]
    assert p == nx.single_source_shortest_path(self.cycle, 0)