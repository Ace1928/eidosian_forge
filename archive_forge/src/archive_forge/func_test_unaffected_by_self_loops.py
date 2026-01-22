import pytest
import networkx as nx
def test_unaffected_by_self_loops(self):
    graph = two_node_graph()
    graph.add_edge(1, 1)
    graph.add_edge(2, 2)
    clique, weight = nx.algorithms.max_weight_clique(graph, 'weight')
    assert verify_clique(graph, clique, weight, 30, 'weight')
    graph = three_node_independent_set()
    graph.add_edge(1, 1)
    clique, weight = nx.algorithms.max_weight_clique(graph, 'weight')
    assert verify_clique(graph, clique, weight, 20, 'weight')