import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_tadpole_graph_for_multigraph(self):
    G = nx.tadpole_graph(5, 20)
    MG = nx.tadpole_graph(5, 20, create_using=nx.MultiGraph)
    assert edges_equal(MG.edges(), G.edges())