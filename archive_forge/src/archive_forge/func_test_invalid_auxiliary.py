import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.utils import pairwise
def test_invalid_auxiliary():
    with pytest.raises(nx.NetworkXError):
        G = nx.complete_graph(5)
        list(nx.node_disjoint_paths(G, 0, 3, auxiliary=G))