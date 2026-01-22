import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_christofides_hamiltonian():
    random.seed(42)
    G = nx.complete_graph(20)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(0, 10)
    H = nx.Graph()
    H.add_edges_from(pairwise(nx_app.christofides(G)))
    H.remove_edges_from(nx.find_cycle(H))
    assert len(H.edges) == 0
    tree = nx.minimum_spanning_tree(G, weight='weight')
    H = nx.Graph()
    H.add_edges_from(pairwise(nx_app.christofides(G, tree)))
    H.remove_edges_from(nx.find_cycle(H))
    assert len(H.edges) == 0