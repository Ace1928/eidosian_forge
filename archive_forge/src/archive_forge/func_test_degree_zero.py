import pytest
import networkx as nx
def test_degree_zero(self):
    """Tests that a degree sequence of all zeros yields the empty
        graph.

        """
    G = nx.configuration_model([0, 0, 0])
    assert len(G) == 3
    assert G.number_of_edges() == 0