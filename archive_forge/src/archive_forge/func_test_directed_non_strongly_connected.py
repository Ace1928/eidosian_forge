import pytest
import networkx as nx
from networkx.algorithms.approximation import diameter
def test_directed_non_strongly_connected(self):
    """Test a directed non strongly connected graph."""
    graph = nx.path_graph(10, create_using=nx.DiGraph())
    with pytest.raises(nx.NetworkXError, match='DiGraph not strongly connected.'):
        diameter(graph)