import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal
def test_format_keyword(self):
    with pytest.raises(nx.NetworkXError):
        bipartite.biadjacency_matrix(nx.Graph([(1, 0)]), [0], format='foo')