import pytest
import networkx as nx
def test_directed_cycle_numpy():
    G = nx.DiGraph()
    nx.add_cycle(G, [0, 1, 2, 3])
    pred, dist = nx.floyd_warshall_predecessor_and_distance(G)
    D = nx.utils.dict_to_numpy_array(dist)
    np.testing.assert_equal(nx.floyd_warshall_numpy(G), D)