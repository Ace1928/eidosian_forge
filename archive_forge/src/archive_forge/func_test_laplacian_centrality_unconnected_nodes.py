import pytest
import networkx as nx
def test_laplacian_centrality_unconnected_nodes():
    """laplacian_centrality on a unconnected node graph should return 0

    For graphs without edges, the Laplacian energy is 0 and is unchanged with
    node removal, so::

        LC(v) = LE(G) - LE(G - v) = 0 - 0 = 0
    """
    G = nx.empty_graph(3)
    assert nx.laplacian_centrality(G, normalized=False) == {0: 0, 1: 0, 2: 0}