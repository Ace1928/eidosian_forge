import pytest
import networkx as nx
from networkx.algorithms import node_classification
def test_nodes_all_labeled(self):
    G = nx.karate_club_graph()
    label_name = 'club'
    predicted = node_classification.local_and_global_consistency(G, alpha=0, label_name=label_name)
    for i in range(len(G)):
        assert predicted[i] == G.nodes[i][label_name]