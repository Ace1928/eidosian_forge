import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_identity_graph_matrix(self):
    """Conversion from graph to sparse matrix to graph."""
    A = nx.to_scipy_sparse_array(self.G1)
    self.identity_conversion(self.G1, A, nx.Graph())