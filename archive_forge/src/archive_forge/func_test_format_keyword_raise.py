import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_format_keyword_raise(self):
    with pytest.raises(nx.NetworkXError):
        WP4 = nx.Graph()
        WP4.add_edges_from(((n, n + 1, {'weight': 0.5, 'other': 0.3}) for n in range(3)))
        P4 = path_graph(4)
        nx.to_scipy_sparse_array(P4, format='any_other')