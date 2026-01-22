import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize(('m', 'n'), [(2, 0), (2, 5), (2, 10), ('ab', 20)])
def test_tadpole_graph_same_as_path_when_m_is_2(self, m, n):
    G = nx.tadpole_graph(m, n)
    assert is_isomorphic(G, nx.path_graph(n + 2))