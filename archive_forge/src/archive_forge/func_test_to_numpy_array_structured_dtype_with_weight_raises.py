import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_to_numpy_array_structured_dtype_with_weight_raises():
    """Using both a structured dtype (with named fields) and specifying a `weight`
    parameter is ambiguous."""
    G = nx.path_graph(3)
    dtype = np.dtype([('weight', int), ('cost', int)])
    exception_msg = 'Specifying `weight` not supported for structured dtypes'
    with pytest.raises(ValueError, match=exception_msg):
        nx.to_numpy_array(G, dtype=dtype)
    with pytest.raises(ValueError, match=exception_msg):
        nx.to_numpy_array(G, dtype=dtype, weight='cost')