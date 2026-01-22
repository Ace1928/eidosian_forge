from itertools import chain, combinations, product
import pytest
import networkx as nx
def assert_lca_dicts_same(self, d1, d2, G=None):
    """Checks if d1 and d2 contain the same pairs and
        have a node at the same distance from root for each.
        If G is None use self.DG."""
    if G is None:
        G = self.DG
        root_distance = self.root_distance
    else:
        roots = [n for n, deg in G.in_degree if deg == 0]
        assert len(roots) == 1
        root_distance = nx.shortest_path_length(G, source=roots[0])
    for a, b in ((min(pair), max(pair)) for pair in chain(d1, d2)):
        assert root_distance[get_pair(d1, a, b)] == root_distance[get_pair(d2, a, b)]