import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
def test_number_of_cliques(self):
    G = self.G
    assert nx.number_of_cliques(G, 1) == 1
    assert list(nx.number_of_cliques(G, [1]).values()) == [1]
    assert list(nx.number_of_cliques(G, [1, 2]).values()) == [1, 2]
    assert nx.number_of_cliques(G, [1, 2]) == {1: 1, 2: 2}
    assert nx.number_of_cliques(G, 2) == 2
    assert nx.number_of_cliques(G) == {1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1}
    assert nx.number_of_cliques(G, nodes=list(G)) == {1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1}
    assert nx.number_of_cliques(G, nodes=[2, 3, 4]) == {2: 2, 3: 1, 4: 2}
    assert nx.number_of_cliques(G, cliques=self.cl) == {1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1}
    assert nx.number_of_cliques(G, list(G), cliques=self.cl) == {1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1}