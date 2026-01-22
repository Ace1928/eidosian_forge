import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_lollipop_graph_right_sizes(self):
    for m1, m2 in [(3, 5), (4, 10), (3, 20)]:
        G = nx.lollipop_graph(m1, m2)
        assert nx.number_of_nodes(G) == m1 + m2
        assert nx.number_of_edges(G) == m1 * (m1 - 1) / 2 + m2
    for first, second in [('ab', ''), ('abc', 'defg')]:
        m1, m2 = (len(first), len(second))
        G = nx.lollipop_graph(first, second)
        assert nx.number_of_nodes(G) == m1 + m2
        assert nx.number_of_edges(G) == m1 * (m1 - 1) / 2 + m2