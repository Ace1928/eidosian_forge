import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
@pytest.mark.parametrize('G', (nx.Graph(), nx.DiGraph()))
def test_to_numpy_array_structured_dtype_scalar_nonedge(G):
    G.add_edge(0, 1, weight=10)
    dtype = np.dtype([('weight', float), ('cost', float)])
    A = nx.to_numpy_array(G, dtype=dtype, weight=None, nonedge=np.nan)
    for attr in dtype.names:
        expected = nx.to_numpy_array(G, dtype=float, weight=attr, nonedge=np.nan)
        npt.assert_array_equal(A[attr], expected)