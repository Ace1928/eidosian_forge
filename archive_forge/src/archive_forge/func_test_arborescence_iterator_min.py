import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_arborescence_iterator_min():
    """
    Tests the arborescence iterator.

    A brute force method found 680 arborescences in this graph.
    This test will not verify all of them individually, but will check two
    things

    * The iterator returns 680 arborescences
    * The weight of the arborescences is non-strictly increasing

    for more information please visit
    https://mjschwenne.github.io/2021/06/10/implementing-the-iterators.html
    """
    G = nx.from_numpy_array(G_array, create_using=nx.DiGraph)
    arborescence_count = 0
    arborescence_weight = -math.inf
    for B in branchings.ArborescenceIterator(G):
        arborescence_count += 1
        new_arborescence_weight = B.size(weight='weight')
        assert new_arborescence_weight >= arborescence_weight
        arborescence_weight = new_arborescence_weight
    assert arborescence_count == 680