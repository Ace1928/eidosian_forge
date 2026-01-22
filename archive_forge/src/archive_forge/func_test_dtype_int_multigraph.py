import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_dtype_int_multigraph(self):
    """Test that setting dtype int actually gives an integer array.

        For more information, see GitHub pull request #1363.

        """
    G = nx.MultiGraph(nx.complete_graph(3))
    A = nx.to_numpy_array(G, dtype=int)
    assert A.dtype == int