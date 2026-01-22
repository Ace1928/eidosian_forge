import networkx as nx
from networkx.algorithms.approximation import (
def test_independent_set():
    G = nx.Graph()
    assert len(maximum_independent_set(G)) == 0