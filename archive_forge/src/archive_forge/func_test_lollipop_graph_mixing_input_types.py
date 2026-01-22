import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_lollipop_graph_mixing_input_types(self):
    cases = [(4, 'abc'), ('abcd', 3), ([1, 2, 3, 4], 'abc'), ('abcd', [1, 2, 3])]
    for m1, m2 in cases:
        G = nx.lollipop_graph(m1, m2)
        assert len(G) == 7
        assert G.size() == 9