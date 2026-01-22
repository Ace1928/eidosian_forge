import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_core_number_multigraph(self):
    G = nx.complete_graph(3)
    G = nx.MultiGraph(G)
    G.add_edge(1, 2)
    with pytest.raises(nx.NetworkXNotImplemented, match='not implemented for multigraph type'):
        nx.core_number(G)