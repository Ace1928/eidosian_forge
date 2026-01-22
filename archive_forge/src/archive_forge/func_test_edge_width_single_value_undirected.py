import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize(('edgewidth', 'expected'), ((3, 3), ([3], 3)))
def test_edge_width_single_value_undirected(edgewidth, expected):
    G = nx.path_graph(4)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos, width=edgewidth)
    assert len(drawn_edges.get_paths()) == 3
    assert drawn_edges.get_linewidth() == expected