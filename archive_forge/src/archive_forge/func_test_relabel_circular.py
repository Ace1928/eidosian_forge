import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_circular(self):
    G = nx.path_graph(3)
    mapping = {0: 1, 1: 0}
    H = nx.relabel_nodes(G, mapping, copy=True)
    with pytest.raises(nx.NetworkXUnfeasible):
        H = nx.relabel_nodes(G, mapping, copy=False)