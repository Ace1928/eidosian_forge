from itertools import combinations
import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_karate_club_graph_cutset(self):
    G = nx.karate_club_graph()
    nx.set_edge_attributes(G, 1, 'capacity')
    T = nx.gomory_hu_tree(G)
    assert nx.is_tree(T)
    u, v = (0, 33)
    cut_value, edge = self.minimum_edge_weight(T, u, v)
    cutset = self.compute_cutset(G, T, edge)
    assert cut_value == len(cutset)