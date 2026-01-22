import itertools
import os
import warnings
import pytest
import networkx as nx
def test_nonzero_selfloop_with_single_edge_in_edgelist():
    """Ensure that selfloop extent is non-zero when only a single edge is
    specified in the edgelist.
    """
    fig, ax = plt.subplots()
    G = nx.path_graph(2, create_using=nx.DiGraph)
    G.add_edge(1, 1)
    pos = {n: (n, n) for n in G.nodes}
    patch = nx.draw_networkx_edges(G, pos, edgelist=[(1, 1)])[0]
    bbox = patch.get_extents()
    assert bbox.width > 0 and bbox.height > 0
    plt.delaxes(ax)