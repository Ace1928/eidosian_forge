import pytest
import networkx as nx
def test_reconstruct_path(self):
    with pytest.raises(KeyError):
        XG = nx.DiGraph()
        XG.add_weighted_edges_from([('s', 'u', 10), ('s', 'x', 5), ('u', 'v', 1), ('u', 'x', 2), ('v', 'y', 1), ('x', 'u', 3), ('x', 'v', 5), ('x', 'y', 2), ('y', 's', 7), ('y', 'v', 6)])
        predecessors, _ = nx.floyd_warshall_predecessor_and_distance(XG)
        path = nx.reconstruct_path('s', 'v', predecessors)
        assert path == ['s', 'x', 'u', 'v']
        path = nx.reconstruct_path('s', 's', predecessors)
        assert path == []
        nx.reconstruct_path('1', '2', predecessors)