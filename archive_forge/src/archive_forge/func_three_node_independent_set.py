import pytest
import networkx as nx
def three_node_independent_set():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3])
    graph.nodes[1]['weight'] = 10
    graph.nodes[2]['weight'] = 20
    graph.nodes[3]['weight'] = 5
    return graph