import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_p(self):
    """Tests for providing an alternate distance metric to the generator."""

    def dist(x, y):
        return sum((abs(a - b) for a, b in zip(x, y)))
    G = nx.thresholded_random_geometric_graph(50, 0.25, 0.1, p=1, seed=42)
    for u, v in combinations(G, 2):
        if v in G[u]:
            assert dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25