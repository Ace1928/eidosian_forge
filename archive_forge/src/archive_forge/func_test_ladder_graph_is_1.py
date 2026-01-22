import pytest
import networkx as nx
def test_ladder_graph_is_1():
    G = nx.ladder_graph(3)
    assert nx.bipartite.maximal_extendability(G) == 1