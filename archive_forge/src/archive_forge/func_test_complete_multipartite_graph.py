import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_complete_multipartite_graph(self):
    """Tests for generating the complete multipartite graph."""
    G = nx.complete_multipartite_graph(2, 3, 4)
    blocks = [(0, 1), (2, 3, 4), (5, 6, 7, 8)]
    for block in blocks:
        for u, v in itertools.combinations_with_replacement(block, 2):
            assert v not in G[u]
            assert G.nodes[u] == G.nodes[v]
    for block1, block2 in itertools.combinations(blocks, 2):
        for u, v in itertools.product(block1, block2):
            assert v in G[u]
            assert G.nodes[u] != G.nodes[v]
    with pytest.raises(nx.NetworkXError, match='Negative number of nodes'):
        nx.complete_multipartite_graph(2, -3, 4)