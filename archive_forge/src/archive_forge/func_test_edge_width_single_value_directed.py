import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize(('edgewidth', 'expected'), ((3, 3), ([3], 3)))
def test_edge_width_single_value_directed(edgewidth, expected):
    G = nx.path_graph(4, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos, width=edgewidth)
    assert len(drawn_edges) == 3
    for fap in drawn_edges:
        assert fap.get_linewidth() == expected