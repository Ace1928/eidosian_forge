import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize('style', ('dashed', '--', (1, (1, 1))))
def test_directed_edges_linestyle_single_value(style):
    """Tests support for specifying linestyles with a single value to be applied to
    all edges in ``draw_networkx_edges`` for FancyArrowPatch outputs
    (e.g. directed edges)."""
    G = nx.path_graph(4, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos, style=style)
    assert len(drawn_edges) == 3
    for fap in drawn_edges:
        assert fap.get_linestyle() == style