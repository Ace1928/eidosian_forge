import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_null_raise(self):
    with pytest.raises(nx.NetworkXError):
        nx.to_scipy_sparse_array(nx.Graph())