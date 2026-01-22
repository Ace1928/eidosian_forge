import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_simple_graphs(self):
    for dest, source in [(to_dict_of_dicts, from_dict_of_dicts), (to_dict_of_lists, from_dict_of_lists)]:
        G = barbell_graph(10, 3)
        G.graph = {}
        dod = dest(G)
        GG = source(dod)
        assert graphs_equal(G, GG)
        GW = to_networkx_graph(dod)
        assert graphs_equal(G, GW)
        GI = nx.Graph(dod)
        assert graphs_equal(G, GI)
        P4 = nx.path_graph(4)
        P3 = nx.path_graph(3)
        P4.graph = {}
        P3.graph = {}
        dod = dest(P4, nodelist=[0, 1, 2])
        Gdod = nx.Graph(dod)
        assert graphs_equal(Gdod, P3)