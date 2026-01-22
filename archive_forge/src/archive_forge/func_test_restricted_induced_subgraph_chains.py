import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_restricted_induced_subgraph_chains(self):
    """Test subgraph chains that both restrict and show nodes/edges.

        A restricted_view subgraph should allow induced subgraphs using
        G.subgraph that automagically without a chain (meaning the result
        is a subgraph view of the original graph not a subgraph-of-subgraph.
        """
    hide_nodes = [3, 4, 5]
    hide_edges = [(6, 7)]
    RG = nx.restricted_view(self.G, hide_nodes, hide_edges)
    nodes = [4, 5, 6, 7, 8]
    SG = nx.induced_subgraph(RG, nodes)
    SSG = RG.subgraph(nodes)
    assert RG._graph is self.G
    assert SSG._graph is self.G
    assert SG._graph is RG
    assert edges_equal(SG.edges, SSG.edges)
    CG = self.G.copy()
    CG.remove_nodes_from(hide_nodes)
    CG.remove_edges_from(hide_edges)
    assert edges_equal(CG.edges(nodes), SSG.edges)
    CG.remove_nodes_from([0, 1, 2, 3])
    assert edges_equal(CG.edges, SSG.edges)
    SSSG = self.G.subgraph(nodes)
    RSG = nx.restricted_view(SSSG, hide_nodes, hide_edges)
    assert RSG._graph is not self.G
    assert edges_equal(RSG.edges, CG.edges)