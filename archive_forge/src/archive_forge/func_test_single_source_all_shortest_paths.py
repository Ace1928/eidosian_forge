import pytest
import networkx as nx
def test_single_source_all_shortest_paths(self):
    cycle_ans = {0: [[0]], 1: [[0, 1]], 2: [[0, 1, 2], [0, 3, 2]], 3: [[0, 3]]}
    ans = dict(nx.single_source_all_shortest_paths(nx.cycle_graph(4), 0))
    assert sorted(ans[2]) == cycle_ans[2]
    ans = dict(nx.single_source_all_shortest_paths(self.grid, 1))
    grid_ans = [[1, 2, 3, 7, 11], [1, 2, 6, 7, 11], [1, 2, 6, 10, 11], [1, 5, 6, 7, 11], [1, 5, 6, 10, 11], [1, 5, 9, 10, 11]]
    assert sorted(ans[11]) == grid_ans
    ans = dict(nx.single_source_all_shortest_paths(nx.cycle_graph(4), 0, weight='weight'))
    assert sorted(ans[2]) == cycle_ans[2]
    ans = dict(nx.single_source_all_shortest_paths(nx.cycle_graph(4), 0, method='bellman-ford', weight='weight'))
    assert sorted(ans[2]) == cycle_ans[2]
    ans = dict(nx.single_source_all_shortest_paths(self.grid, 1, weight='weight'))
    assert sorted(ans[11]) == grid_ans
    ans = dict(nx.single_source_all_shortest_paths(self.grid, 1, method='bellman-ford', weight='weight'))
    assert sorted(ans[11]) == grid_ans
    G = nx.cycle_graph(4)
    G.add_node(4)
    ans = dict(nx.single_source_all_shortest_paths(G, 0))
    assert sorted(ans[2]) == [[0, 1, 2], [0, 3, 2]]
    ans = dict(nx.single_source_all_shortest_paths(G, 4))
    assert sorted(ans[4]) == [[4]]