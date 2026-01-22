import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_identity_weighted_digraph_matrix(self):
    """Conversion from weighted digraph to sparse matrix to weighted digraph."""
    A = nx.to_scipy_sparse_array(self.G4)
    self.identity_conversion(self.G4, A, nx.DiGraph())