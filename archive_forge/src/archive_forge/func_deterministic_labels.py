import pytest
import networkx as nx
def deterministic_labels(self, G):
    node_labels = list(G.nodes)
    node_labels = sorted(node_labels, key=lambda n: sorted(G.nodes[n]['group'])[0])
    node_labels.sort()
    label_mapping = {}
    for index, node in enumerate(node_labels):
        label = 'Supernode-%s' % index
        label_mapping[node] = label
    return nx.relabel_nodes(G, label_mapping)