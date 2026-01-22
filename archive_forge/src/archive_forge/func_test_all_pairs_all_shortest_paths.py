import pytest
import networkx as nx
def test_all_pairs_all_shortest_paths(self):
    ans = dict(nx.all_pairs_all_shortest_paths(nx.cycle_graph(4)))
    assert sorted(ans[1][3]) == [[1, 0, 3], [1, 2, 3]]
    ans = dict(nx.all_pairs_all_shortest_paths(nx.cycle_graph(4)), weight='weight')
    assert sorted(ans[1][3]) == [[1, 0, 3], [1, 2, 3]]
    ans = dict(nx.all_pairs_all_shortest_paths(nx.cycle_graph(4)), method='bellman-ford', weight='weight')
    assert sorted(ans[1][3]) == [[1, 0, 3], [1, 2, 3]]
    G = nx.cycle_graph(4)
    G.add_node(4)
    ans = dict(nx.all_pairs_all_shortest_paths(G))
    assert sorted(ans[4][4]) == [[4]]