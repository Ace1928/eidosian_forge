import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_to_edgelist(self):
    G = nx.Graph([(1, 1)])
    elist = nx.to_edgelist(G, nodelist=list(G))
    assert edges_equal(G.edges(data=True), elist)