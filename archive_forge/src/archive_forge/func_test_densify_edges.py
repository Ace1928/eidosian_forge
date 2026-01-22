import pytest
import networkx as nx
def test_densify_edges(self):
    """
        Verifies that densification produces the correct edges from the
        original directed graph
        """
    compressed_G = self.build_compressed_graph()
    original_graph = self.densify(compressed_G, self.c_nodes, copy=True)
    G = self.build_original_graph()
    for s, t in G.edges():
        assert G.has_edge(s, t) == original_graph.has_edge(s, t)