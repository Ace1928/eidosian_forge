import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_random_spanning_tree_single_node_graph():
    G = nx.Graph()
    G.add_node(0)
    rst = nx.tree.random_spanning_tree(G)
    assert len(rst.nodes) == 1
    assert len(rst.edges) == 0