import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
@pytest.mark.parametrize(('field_name', 'expected_attr_val'), [('weight', 1), ('cost', 3)])
def test_to_numpy_array_structured_dtype_single_attr(field_name, expected_attr_val):
    G = nx.Graph()
    G.add_edge(0, 1, cost=3)
    dtype = np.dtype([(field_name, float)])
    A = nx.to_numpy_array(G, dtype=dtype, weight=None)
    expected = np.array([[0, expected_attr_val], [expected_attr_val, 0]], dtype=float)
    npt.assert_array_equal(A[field_name], expected)