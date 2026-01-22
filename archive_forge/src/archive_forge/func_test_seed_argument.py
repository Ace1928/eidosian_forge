import itertools
import pytest
import networkx as nx
def test_seed_argument(self):
    graph = lf_shc()
    rs = nx.coloring.strategy_random_sequential
    c1 = nx.coloring.greedy_color(graph, lambda g, c: rs(g, c, seed=1))
    for u, v in graph.edges:
        assert c1[u] != c1[v]