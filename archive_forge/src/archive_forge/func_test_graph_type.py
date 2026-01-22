from itertools import chain, combinations
import pytest
import networkx as nx
def test_graph_type(self):
    G1 = nx.complete_graph(self.N, nx.MultiDiGraph())
    G2 = nx.MultiGraph(G1)
    G3 = nx.DiGraph(G1)
    G4 = nx.Graph(G1)
    truth = {frozenset(G1)}
    self._check_communities(G1, truth)
    self._check_communities(G2, truth)
    self._check_communities(G3, truth)
    self._check_communities(G4, truth)