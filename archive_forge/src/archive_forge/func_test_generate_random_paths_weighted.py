import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_generate_random_paths_weighted(self):
    np.random.seed(42)
    index_map = {}
    num_paths = 10
    path_length = 6
    G = nx.Graph()
    G.add_edge('a', 'b', weight=0.6)
    G.add_edge('a', 'c', weight=0.2)
    G.add_edge('c', 'd', weight=0.1)
    G.add_edge('c', 'e', weight=0.7)
    G.add_edge('c', 'f', weight=0.9)
    G.add_edge('a', 'd', weight=0.3)
    paths = nx.generate_random_paths(G, num_paths, path_length=path_length, index_map=index_map)
    expected_paths = [['d', 'c', 'f', 'c', 'd', 'a', 'b'], ['e', 'c', 'f', 'c', 'f', 'c', 'e'], ['d', 'a', 'b', 'a', 'b', 'a', 'c'], ['b', 'a', 'd', 'a', 'b', 'a', 'b'], ['d', 'a', 'b', 'a', 'b', 'a', 'd'], ['d', 'a', 'b', 'a', 'b', 'a', 'c'], ['d', 'a', 'b', 'a', 'b', 'a', 'b'], ['f', 'c', 'f', 'c', 'f', 'c', 'e'], ['d', 'a', 'd', 'a', 'b', 'a', 'b'], ['e', 'c', 'f', 'c', 'e', 'c', 'd']]
    expected_map = {'d': {0, 2, 3, 4, 5, 6, 8, 9}, 'c': {0, 1, 2, 5, 7, 9}, 'f': {0, 1, 9, 7}, 'a': {0, 2, 3, 4, 5, 6, 8}, 'b': {0, 2, 3, 4, 5, 6, 8}, 'e': {1, 9, 7}}
    assert expected_paths == list(paths)
    assert expected_map == index_map