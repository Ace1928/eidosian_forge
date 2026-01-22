import pytest
import networkx as nx
def test_numpy_degree_sequence():
    np = pytest.importorskip('numpy')
    ds = np.array([1, 2, 2, 2, 1], dtype=np.int64)
    assert nx.is_graphical(ds, 'eg')
    assert nx.is_graphical(ds, 'hh')
    ds = np.array([1, 2, 2, 2, 1], dtype=np.float64)
    assert nx.is_graphical(ds, 'eg')
    assert nx.is_graphical(ds, 'hh')
    ds = np.array([1.1, 2, 2, 2, 1], dtype=np.float64)
    pytest.raises(nx.NetworkXException, nx.is_graphical, ds, 'eg')
    pytest.raises(nx.NetworkXException, nx.is_graphical, ds, 'hh')