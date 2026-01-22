import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize('style_seq', (['dashed'], ['--'], [(1, (1, 1))], ['--', '-', ':'], ['--', '-'], ['--', '-', ':', '-.']))
def test_directed_edges_linestyle_sequence(style_seq):
    """Tests support for specifying linestyles with sequences in
    ``draw_networkx_edges`` for FancyArrowPatch outputs (e.g. directed edges)."""
    G = nx.path_graph(4, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos, style=style_seq)
    assert len(drawn_edges) == 3
    for fap, style in zip(drawn_edges, itertools.cycle(style_seq)):
        assert fap.get_linestyle() == style