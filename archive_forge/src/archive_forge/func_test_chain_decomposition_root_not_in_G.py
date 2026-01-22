from itertools import cycle, islice
import pytest
import networkx as nx
def test_chain_decomposition_root_not_in_G(self):
    """Test chain decomposition when root is not in graph"""
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    with pytest.raises(nx.NodeNotFound):
        nx.has_bridges(G, root=6)