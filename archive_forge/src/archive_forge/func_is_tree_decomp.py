import itertools
import networkx as nx
from networkx.algorithms.approximation import (
from networkx.algorithms.approximation.treewidth import (
def is_tree_decomp(graph, decomp):
    """Check if the given tree decomposition is valid."""
    for x in graph.nodes():
        appear_once = False
        for bag in decomp.nodes():
            if x in bag:
                appear_once = True
                break
        assert appear_once
    for x, y in graph.edges():
        appear_together = False
        for bag in decomp.nodes():
            if x in bag and y in bag:
                appear_together = True
                break
        assert appear_together
    for v in graph.nodes():
        subset = []
        for bag in decomp.nodes():
            if v in bag:
                subset.append(bag)
        sub_graph = decomp.subgraph(subset)
        assert nx.is_connected(sub_graph)