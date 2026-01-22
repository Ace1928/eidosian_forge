import collections
import pytest
import networkx as nx
@pytest.mark.parametrize(('graph_type', 'result'), ((nx.MultiGraph, [(0, 1, 0), (1, 0, 1)]), (nx.MultiDiGraph, [(0, 1, 0), (1, 0, 0)])))
def test_eulerian_with_keys(self, graph_type, result):
    G = graph_type([(0, 1), (1, 0)])
    answer = nx.eulerian_path(G, keys=True)
    assert list(answer) == result