import itertools
import os
import warnings
import pytest
import networkx as nx
def test_draw_empty_nodes_return_values():
    import matplotlib.collections
    G = nx.Graph([(1, 2), (2, 3)])
    DG = nx.DiGraph([(1, 2), (2, 3)])
    pos = nx.circular_layout(G)
    assert isinstance(nx.draw_networkx_nodes(G, pos, nodelist=[]), mpl.collections.PathCollection)
    assert isinstance(nx.draw_networkx_nodes(DG, pos, nodelist=[]), mpl.collections.PathCollection)
    assert nx.draw_networkx_edges(G, pos, edgelist=[], arrows=True) == []
    assert nx.draw_networkx_edges(G, pos, edgelist=[], arrows=False) == []
    assert nx.draw_networkx_edges(DG, pos, edgelist=[], arrows=False) == []
    assert nx.draw_networkx_edges(DG, pos, edgelist=[], arrows=True) == []