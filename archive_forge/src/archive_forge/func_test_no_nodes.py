import pytest
import networkx as nx
from networkx.algorithms import node_classification
def test_no_nodes(self):
    with pytest.raises(nx.NetworkXError):
        G = nx.Graph()
        node_classification.local_and_global_consistency(G)