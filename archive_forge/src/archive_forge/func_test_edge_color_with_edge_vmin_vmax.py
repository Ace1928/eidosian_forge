import itertools
import os
import warnings
import pytest
import networkx as nx
def test_edge_color_with_edge_vmin_vmax():
    """Test that edge_vmin and edge_vmax properly set the dynamic range of the
    color map when num edges == len(edge_colors)."""
    G = nx.path_graph(3, create_using=nx.DiGraph)
    pos = nx.random_layout(G)
    drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=[0, 1.0])
    orig_colors = [e.get_edgecolor() for e in drawn_edges]
    drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=[0.2, 0.8], edge_vmin=0.2, edge_vmax=0.8)
    scaled_colors = [e.get_edgecolor() for e in drawn_edges]
    assert mpl.colors.same_color(orig_colors, scaled_colors)