import random
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
def test_random_labeled_rooted_tree():
    for i in range(1, 10):
        t1 = nx.random_labeled_rooted_tree(i, seed=42)
        t2 = nx.random_labeled_rooted_tree(i, seed=42)
        assert nx.utils.misc.graphs_equal(t1, t2)
        assert nx.is_tree(t1)
        assert 'root' in t1.graph
        assert 'roots' not in t1.graph