import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize('arrowsize', (30, [10, 20, 30]))
def test_draw_edges_arrowsize(arrowsize):
    G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    pos = {0: (0, 0), 1: (0, 1), 2: (1, 0)}
    edges = nx.draw_networkx_edges(G, pos=pos, arrowsize=arrowsize)
    arrowsize = itertools.repeat(arrowsize) if isinstance(arrowsize, int) else arrowsize
    for fap, expected in zip(edges, arrowsize):
        assert isinstance(fap, mpl.patches.FancyArrowPatch)
        assert fap.get_mutation_scale() == expected