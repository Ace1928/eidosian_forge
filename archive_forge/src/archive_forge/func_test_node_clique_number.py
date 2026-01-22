import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
def test_node_clique_number(self):
    G = self.G
    assert nx.node_clique_number(G, 1) == 4
    assert list(nx.node_clique_number(G, [1]).values()) == [4]
    assert list(nx.node_clique_number(G, [1, 2]).values()) == [4, 4]
    assert nx.node_clique_number(G, [1, 2]) == {1: 4, 2: 4}
    assert nx.node_clique_number(G, 1) == 4
    assert nx.node_clique_number(G) == {1: 4, 2: 4, 3: 4, 4: 3, 5: 3, 6: 4, 7: 3, 8: 2, 9: 2, 10: 2, 11: 2}
    assert nx.node_clique_number(G, cliques=self.cl) == {1: 4, 2: 4, 3: 4, 4: 3, 5: 3, 6: 4, 7: 3, 8: 2, 9: 2, 10: 2, 11: 2}
    assert nx.node_clique_number(G, [1, 2], cliques=self.cl) == {1: 4, 2: 4}
    assert nx.node_clique_number(G, 1, cliques=self.cl) == 4