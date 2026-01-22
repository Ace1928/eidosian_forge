import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
@pytest.mark.parametrize('G, expected', [(nx.Graph(), np.array([[0, 1 + 2j], [1 + 2j, 0]], dtype=complex)), (nx.DiGraph(), np.array([[0, 1 + 2j], [0, 0]], dtype=complex))])
def test_to_numpy_array_complex_weights(G, expected):
    G.add_edge(0, 1, weight=1 + 2j)
    A = nx.to_numpy_array(G, dtype=complex)
    npt.assert_array_equal(A, expected)