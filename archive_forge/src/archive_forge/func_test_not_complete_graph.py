import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_not_complete_graph(self):
    pytest.raises(nx.NetworkXError, self.tsp, self.incompleteUG, 'greedy', source=0)
    pytest.raises(nx.NetworkXError, self.tsp, self.incompleteDG, 'greedy', source=0)