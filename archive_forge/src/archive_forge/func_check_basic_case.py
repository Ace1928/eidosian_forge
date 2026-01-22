import itertools
import pytest
import networkx as nx
def check_basic_case(graph_func, n_nodes, strategy, interchange):
    graph = graph_func()
    coloring = nx.coloring.greedy_color(graph, strategy=strategy, interchange=interchange)
    assert verify_length(coloring, n_nodes)
    assert verify_coloring(graph, coloring)