import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_turan_graph(self):
    assert nx.number_of_edges(nx.turan_graph(13, 4)) == 63
    assert is_isomorphic(nx.turan_graph(13, 4), nx.complete_multipartite_graph(3, 4, 3, 3))