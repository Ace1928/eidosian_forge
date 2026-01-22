import pytest
import networkx as nx
def test_trophic_levels_levine():
    """Example from Figure 5 in Stephen Levine (1980) J. theor. Biol. 83,
    195-207
    """
    S = nx.DiGraph()
    S.add_edge(1, 2, weight=1.0)
    S.add_edge(1, 3, weight=0.2)
    S.add_edge(1, 4, weight=0.8)
    S.add_edge(2, 3, weight=0.2)
    S.add_edge(2, 5, weight=0.3)
    S.add_edge(4, 3, weight=0.6)
    S.add_edge(4, 5, weight=0.7)
    S.add_edge(5, 4, weight=0.2)
    S2 = S.copy()
    z = [nid for nid, d in S.in_degree if d == 0]
    for nid in z:
        S.remove_node(nid)
    q = nx.linalg.graphmatrix.adjacency_matrix(S).T
    expected_q = np.array([[0, 0, 0.0, 0], [0.2, 0, 0.6, 0], [0, 0, 0, 0.2], [0.3, 0, 0.7, 0]])
    assert np.array_equal(q.todense(), expected_q)
    assert len(q.shape) == 2
    assert q.shape[0] == q.shape[1]
    assert q.shape[0] == len(S)
    nn = q.shape[0]
    i = np.eye(nn)
    n = np.linalg.inv(i - q)
    y = np.asarray(n) @ np.ones(nn)
    expected_y = np.array([1, 2.07906977, 1.46511628, 2.3255814])
    assert np.allclose(y, expected_y)
    expected_d = {1: 1, 2: 2, 3: 3.07906977, 4: 2.46511628, 5: 3.3255814}
    d = nx.trophic_levels(S2)
    for nid, level in d.items():
        expected_level = expected_d[nid]
        assert expected_level == pytest.approx(level, abs=1e-07)