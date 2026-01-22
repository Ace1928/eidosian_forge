import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_identity_weighted_graph_matrix(self):
    """Conversion from weighted graph to sparse matrix to weighted graph."""
    A = nx.to_scipy_sparse_array(self.G3)
    self.identity_conversion(self.G3, A, nx.Graph())