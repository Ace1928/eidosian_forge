import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_is_triad():
    """Tests the is_triad function"""
    G = nx.karate_club_graph()
    G = G.to_directed()
    for i in range(100):
        nodes = sample(sorted(G.nodes()), 3)
        G2 = G.subgraph(nodes)
        assert nx.is_triad(G2)