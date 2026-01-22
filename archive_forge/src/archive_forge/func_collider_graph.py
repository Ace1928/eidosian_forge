from itertools import combinations
import pytest
import networkx as nx
def collider_graph():
    """Return a collider/v-structure graph with three nodes."""
    G = nx.DiGraph(name='collider')
    G.add_edges_from([(0, 2), (1, 2)])
    nx.freeze(G)
    return G