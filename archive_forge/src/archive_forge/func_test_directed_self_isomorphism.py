import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_directed_self_isomorphism(self):
    """
        For some small, directed, symmetric graphs, make sure that 1) they are
        isomorphic to themselves, and 2) that only the identity mapping is
        found.
        """
    for node_data, edge_data in self.data:
        graph = nx.Graph()
        graph.add_nodes_from(node_data)
        graph.add_edges_from(edge_data)
        ismags = iso.ISMAGS(graph, graph, node_match=iso.categorical_node_match('name', None))
        assert ismags.is_isomorphic()
        assert ismags.subgraph_is_isomorphic()
        assert list(ismags.subgraph_isomorphisms_iter(symmetry=True)) == [{n: n for n in graph.nodes}]