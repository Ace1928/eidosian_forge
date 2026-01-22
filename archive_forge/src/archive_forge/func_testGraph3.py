import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def testGraph3(self):
    G1 = getCanonical()
    G2 = nx.Graph()
    G2.add_node('A', label='A')
    G2.add_node('B', label='B')
    G2.add_node('C', label='C')
    G2.add_node('D', label='D')
    G2.add_node('E', label='E')
    G2.add_node('F', label='F')
    G2.add_node('G', label='G')
    G2.add_edge('A', 'C', label='a-c')
    G2.add_edge('A', 'D', label='a-d')
    G2.add_edge('D', 'E', label='d-e')
    G2.add_edge('D', 'F', label='d-f')
    G2.add_edge('D', 'G', label='d-g')
    G2.add_edge('E', 'B', label='e-b')
    assert graph_edit_distance(G1, G2, node_match=nmatch, edge_match=ematch) == 12