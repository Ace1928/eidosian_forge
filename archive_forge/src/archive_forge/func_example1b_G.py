import pytest
import networkx as nx
def example1b_G():
    G = nx.Graph()
    G.add_node(1, percolation=0.3)
    G.add_node(2, percolation=0.5)
    G.add_node(3, percolation=0.5)
    G.add_node(4, percolation=0.2)
    G.add_node(5, percolation=0.3)
    G.add_node(6, percolation=0.2)
    G.add_node(7, percolation=0.1)
    G.add_node(8, percolation=0.1)
    G.add_edges_from([(1, 4), (2, 4), (3, 4), (4, 5), (5, 6), (6, 7), (6, 8)])
    return G