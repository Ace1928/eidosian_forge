import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_arborescence_iterator_initial_partition():
    """
    Tests the arborescence iterator with three included edges and three excluded
    in the initial partition.

    A brute force method similar to the one used in the above tests found that
    there are 16 arborescences which contain the included edges and not the
    excluded edges.
    """
    G = nx.from_numpy_array(G_array, create_using=nx.DiGraph)
    included_edges = [(1, 0), (5, 6), (8, 7)]
    excluded_edges = [(0, 2), (3, 6), (1, 5)]
    arborescence_count = 0
    arborescence_weight = -math.inf
    for B in branchings.ArborescenceIterator(G, init_partition=(included_edges, excluded_edges)):
        arborescence_count += 1
        new_arborescence_weight = B.size(weight='weight')
        assert new_arborescence_weight >= arborescence_weight
        arborescence_weight = new_arborescence_weight
        for e in included_edges:
            assert e in B.edges
        for e in excluded_edges:
            assert e not in B.edges
    assert arborescence_count == 16