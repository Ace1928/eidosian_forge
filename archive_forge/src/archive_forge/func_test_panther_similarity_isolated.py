import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_panther_similarity_isolated(self):
    G = nx.Graph()
    G.add_nodes_from(range(5))
    with pytest.raises(nx.NetworkXUnfeasible, match='Panther similarity is not defined for the isolated source node 1.'):
        nx.panther_similarity(G, source=1)