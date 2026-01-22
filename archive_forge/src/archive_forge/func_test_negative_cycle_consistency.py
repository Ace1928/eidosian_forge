import pytest
import networkx as nx
from networkx.utils import pairwise
def test_negative_cycle_consistency(self):
    import random
    unif = random.uniform
    for random_seed in range(2):
        random.seed(random_seed)
        for density in [0.1, 0.9]:
            for N in [1, 10, 20]:
                for max_cost in [1, 90]:
                    G = nx.binomial_graph(N, density, seed=4, directed=True)
                    edges = ((u, v, unif(-1, max_cost)) for u, v in G.edges)
                    G.add_weighted_edges_from(edges)
                    no_heuristic = nx.negative_edge_cycle(G, heuristic=False)
                    with_heuristic = nx.negative_edge_cycle(G, heuristic=True)
                    assert no_heuristic == with_heuristic