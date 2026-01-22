from itertools import chain, combinations
import pytest
import networkx as nx
def test_random_graph(self):
    G = nx.gnm_random_graph(self.N, self.N * self.K // 2)
    truth = {frozenset(G)}
    self._check_communities(G, truth)