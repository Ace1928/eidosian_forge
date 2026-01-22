import itertools
import pytest
import networkx as nx
def test_strategy_as_function(self):
    graph = lf_shc()
    colors_1 = nx.coloring.greedy_color(graph, 'largest_first')
    colors_2 = nx.coloring.greedy_color(graph, nx.coloring.strategy_largest_first)
    assert colors_1 == colors_2