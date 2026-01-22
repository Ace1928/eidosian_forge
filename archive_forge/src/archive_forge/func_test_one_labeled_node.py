import pytest
import networkx as nx
from networkx.algorithms import node_classification
def test_one_labeled_node(self):
    G = nx.path_graph(4)
    label_name = 'label'
    G.nodes[0][label_name] = 'A'
    predicted = node_classification.local_and_global_consistency(G, label_name=label_name)
    assert predicted[0] == 'A'
    assert predicted[1] == 'A'
    assert predicted[2] == 'A'
    assert predicted[3] == 'A'