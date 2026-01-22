import itertools
import os
import warnings
import pytest
import networkx as nx
def test_alpha_iter():
    pos = nx.random_layout(barbell)
    fig = plt.figure()
    fig.add_subplot(131)
    nx.draw_networkx_nodes(barbell, pos, alpha=[0.1, 0.2])
    num_nodes = len(barbell.nodes)
    alpha = [x / num_nodes for x in range(num_nodes)]
    colors = range(num_nodes)
    fig.add_subplot(132)
    nx.draw_networkx_nodes(barbell, pos, node_color=colors, alpha=alpha)
    alpha.append(1)
    fig.add_subplot(133)
    nx.draw_networkx_nodes(barbell, pos, alpha=alpha)