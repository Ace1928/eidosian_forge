from itertools import combinations
import pytest
import networkx as nx
def fork_graph():
    """Return a three node fork graph."""
    G = nx.DiGraph(name='fork')
    G.add_edges_from([(0, 1), (0, 2)])
    nx.freeze(G)
    return G