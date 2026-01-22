import pytest
import networkx as nx
def test_has_bridges_raises_root_not_in_G(self):
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    with pytest.raises(nx.NodeNotFound):
        nx.has_bridges(G, root=6)