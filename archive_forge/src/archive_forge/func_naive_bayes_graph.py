from itertools import combinations
import pytest
import networkx as nx
def naive_bayes_graph():
    """Return a simply Naive Bayes PGM graph."""
    G = nx.DiGraph(name='naive_bayes')
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
    nx.freeze(G)
    return G