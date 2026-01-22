import pytest
import networkx as nx
def test_error_on_non_integer_weight(self):
    graph = two_node_graph()
    graph.nodes[2]['weight'] = 1.5
    with pytest.raises(ValueError):
        nx.algorithms.max_weight_clique(graph)