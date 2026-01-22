import pytest
import networkx as nx
from networkx.algorithms.approximation import diameter
def test_complete_directed_graph(self):
    """Test a complete directed graph."""
    graph = nx.complete_graph(10, create_using=nx.DiGraph())
    assert diameter(graph) == 1