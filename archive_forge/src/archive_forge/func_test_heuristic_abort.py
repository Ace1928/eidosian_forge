import itertools
import networkx as nx
from networkx.algorithms.approximation import (
from networkx.algorithms.approximation.treewidth import (
def test_heuristic_abort(self):
    """Test if min_fill_in returns None for fully connected graph"""
    graph = {}
    for u in self.complete:
        graph[u] = set()
        for v in self.complete[u]:
            if u != v:
                graph[u].add(v)
    next_node = min_fill_in_heuristic(graph)
    if next_node is None:
        pass
    else:
        assert False