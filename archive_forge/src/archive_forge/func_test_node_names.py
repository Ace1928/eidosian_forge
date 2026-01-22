import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_node_names(self):
    """Tests using values other than sequential numbers as node IDs."""
    import string
    nodes = list(string.ascii_lowercase)
    G = nx.thresholded_random_geometric_graph(nodes, 0.25, 0.1, seed=42)
    assert len(G) == len(nodes)
    for u, v in combinations(G, 2):
        if v in G[u]:
            assert math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25