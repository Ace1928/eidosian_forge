import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_panther_similarity_unweighted(self):
    np.random.seed(42)
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    G.add_edge(1, 2)
    G.add_edge(2, 4)
    expected = {3: 0.5, 2: 0.5, 1: 0.5, 4: 0.125}
    sim = nx.panther_similarity(G, 0, path_length=2)
    assert sim == expected