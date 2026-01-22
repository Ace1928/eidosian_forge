import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_tadpole_graph_non_builtin_integers(self):
    np = pytest.importorskip('numpy')
    G = nx.tadpole_graph(np.int32(4), np.int64(3))
    expected = nx.compose(nx.cycle_graph(4), nx.path_graph(range(100, 103)))
    expected.add_edge(0, 100)
    assert is_isomorphic(G, expected)