import pytest
import networkx as nx
def test_graph_with_isolated_v(self):
    G = nx.Graph()
    G.add_node(1)
    with pytest.raises(nx.NetworkXException, match='Graph has a node with no edge incident on it, so no edge cover exists.'):
        nx.min_edge_cover(G)