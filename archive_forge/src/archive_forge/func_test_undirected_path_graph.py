import pytest
import networkx as nx
from networkx.algorithms.approximation import diameter
def test_undirected_path_graph(self):
    """Test an undirected path graph with 10 nodes."""
    graph = nx.path_graph(10)
    assert diameter(graph) == 9