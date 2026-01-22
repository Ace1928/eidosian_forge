import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
def test_paper_figure_4(self):
    G = nx.Graph()
    edges_fig_4 = [('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'e'), ('b', 'c'), ('b', 'd'), ('b', 'e'), ('c', 'd'), ('c', 'e'), ('d', 'e'), ('f', 'b'), ('f', 'c'), ('f', 'g'), ('g', 'f'), ('g', 'c'), ('g', 'd'), ('g', 'e')]
    G.add_edges_from(edges_fig_4)
    cliques = list(nx.enumerate_all_cliques(G))
    clique_sizes = list(map(len, cliques))
    assert sorted(clique_sizes) == clique_sizes
    expected_cliques = [['a'], ['b'], ['c'], ['d'], ['e'], ['f'], ['g'], ['a', 'b'], ['a', 'b', 'd'], ['a', 'b', 'd', 'e'], ['a', 'b', 'e'], ['a', 'c'], ['a', 'c', 'd'], ['a', 'c', 'd', 'e'], ['a', 'c', 'e'], ['a', 'd'], ['a', 'd', 'e'], ['a', 'e'], ['b', 'c'], ['b', 'c', 'd'], ['b', 'c', 'd', 'e'], ['b', 'c', 'e'], ['b', 'c', 'f'], ['b', 'd'], ['b', 'd', 'e'], ['b', 'e'], ['b', 'f'], ['c', 'd'], ['c', 'd', 'e'], ['c', 'd', 'e', 'g'], ['c', 'd', 'g'], ['c', 'e'], ['c', 'e', 'g'], ['c', 'f'], ['c', 'f', 'g'], ['c', 'g'], ['d', 'e'], ['d', 'e', 'g'], ['d', 'g'], ['e', 'g'], ['f', 'g'], ['a', 'b', 'c'], ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'e']]
    assert sorted(map(sorted, cliques)) == sorted(map(sorted, expected_cliques))