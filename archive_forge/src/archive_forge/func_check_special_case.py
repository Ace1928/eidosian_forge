import itertools
import pytest
import networkx as nx
def check_special_case(strategy, graph_func, interchange, colors):
    graph = graph_func()
    coloring = nx.coloring.greedy_color(graph, strategy=strategy, interchange=interchange)
    if not hasattr(colors, '__len__'):
        colors = [colors]
    assert any((verify_length(coloring, n_colors) for n_colors in colors))
    assert verify_coloring(graph, coloring)