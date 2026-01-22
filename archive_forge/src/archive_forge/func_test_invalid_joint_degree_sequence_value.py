import pytest
import networkx as nx
def test_invalid_joint_degree_sequence_value(self):
    with pytest.raises(nx.NetworkXError, match='Invalid degree sequence'):
        nx.random_clustered_graph([[1, 1], [1, 2], [0, 1]])