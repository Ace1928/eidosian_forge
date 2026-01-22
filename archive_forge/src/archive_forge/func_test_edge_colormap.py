import itertools
import os
import warnings
import pytest
import networkx as nx
def test_edge_colormap():
    colors = range(barbell.number_of_edges())
    nx.draw_spring(barbell, edge_color=colors, width=4, edge_cmap=plt.cm.Blues, with_labels=True)