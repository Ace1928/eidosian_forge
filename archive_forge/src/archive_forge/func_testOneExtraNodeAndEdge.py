import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def testOneExtraNodeAndEdge(self):
    G1 = nx.Graph()
    G1.add_node('A', label='A')
    G1.add_node('B', label='B')
    G1.add_edge('A', 'B', label='a-b')
    G2 = nx.Graph()
    G2.add_node('A', label='A')
    G2.add_node('B', label='B')
    G2.add_node('C', label='C')
    G2.add_edge('A', 'B', label='a-b')
    G2.add_edge('A', 'C', label='a-c')
    assert graph_edit_distance(G1, G2, node_match=nmatch, edge_match=ematch) == 2