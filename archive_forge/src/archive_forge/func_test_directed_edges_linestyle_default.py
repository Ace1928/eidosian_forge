import itertools
import os
import warnings
import pytest
import networkx as nx
def test_directed_edges_linestyle_default():
    """Test default linestyle for edges drawn with FancyArrowPatches."""
    G = nx.path_graph(4, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos)
    assert len(drawn_edges) == 3
    for fap in drawn_edges:
        assert fap.get_linestyle() == 'solid'