import itertools
import os
import warnings
import pytest
import networkx as nx
def test_fewer_edge_colors_than_num_edges_directed():
    """Test that the edge colors are cycled when there are fewer specified
    colors than edges."""
    G = barbell.to_directed()
    pos = nx.random_layout(barbell)
    edgecolors = ('r', 'g', 'b')
    drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=edgecolors)
    for fap, expected in zip(drawn_edges, itertools.cycle(edgecolors)):
        assert mpl.colors.same_color(fap.get_edgecolor(), expected)