import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_from_numpy_array_type(self):
    A = np.array([[1]])
    G = nx.from_numpy_array(A)
    assert type(G[0][0]['weight']) == int
    A = np.array([[1]]).astype(float)
    G = nx.from_numpy_array(A)
    assert type(G[0][0]['weight']) == float
    A = np.array([[1]]).astype(str)
    G = nx.from_numpy_array(A)
    assert type(G[0][0]['weight']) == str
    A = np.array([[1]]).astype(bool)
    G = nx.from_numpy_array(A)
    assert type(G[0][0]['weight']) == bool
    A = np.array([[1]]).astype(complex)
    G = nx.from_numpy_array(A)
    assert type(G[0][0]['weight']) == complex
    A = np.array([[1]]).astype(object)
    pytest.raises(TypeError, nx.from_numpy_array, A)
    A = np.array([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
    with pytest.raises(nx.NetworkXError, match=f'Input array must be 2D, not {A.ndim}'):
        g = nx.from_numpy_array(A)