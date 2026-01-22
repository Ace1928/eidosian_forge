import itertools
import os
import warnings
import pytest
import networkx as nx
def test_nonzero_selfloop_with_single_node():
    """Ensure that selfloop extent is non-zero when there is only one node."""
    fig, ax = plt.subplots()
    G = nx.DiGraph()
    G.add_node(0)
    G.add_edge(0, 0)
    patch = nx.draw_networkx_edges(G, {0: (0, 0)})[0]
    bbox = patch.get_extents()
    assert bbox.width > 0 and bbox.height > 0
    plt.delaxes(ax)