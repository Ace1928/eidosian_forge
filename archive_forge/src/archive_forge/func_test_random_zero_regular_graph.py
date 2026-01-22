import pytest
import networkx as nx
def test_random_zero_regular_graph(self):
    """Tests that a 0-regular graph has the correct number of nodes and
        edges.

        """
    seed = 42
    G = nx.random_regular_graph(0, 10, seed)
    assert len(G) == 10
    assert G.number_of_edges() == 0