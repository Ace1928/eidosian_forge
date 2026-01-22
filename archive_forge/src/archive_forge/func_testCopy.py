import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def testCopy(self):
    G = nx.Graph()
    G.add_node('A', label='A')
    G.add_node('B', label='B')
    G.add_edge('A', 'B', label='a-b')
    assert graph_edit_distance(G, G.copy(), node_match=nmatch, edge_match=ematch) == 0