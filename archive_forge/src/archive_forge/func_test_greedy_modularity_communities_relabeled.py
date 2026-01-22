import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_greedy_modularity_communities_relabeled():
    G = nx.balanced_tree(2, 2)
    mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    G = nx.relabel_nodes(G, mapping)
    expected = [frozenset({'e', 'd', 'a', 'b'}), frozenset({'c', 'f', 'g'})]
    assert greedy_modularity_communities(G) == expected