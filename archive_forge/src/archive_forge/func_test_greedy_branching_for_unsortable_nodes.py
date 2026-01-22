import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_greedy_branching_for_unsortable_nodes():
    G = nx.DiGraph()
    G.add_weighted_edges_from([((2, 3), 5, 1), (3, 'a', 1), (2, 4, 5)])
    edges = [(u, v, data.get('weight', 1)) for u, v, data in G.edges(data=True)]
    with pytest.raises(TypeError):
        edges.sort(key=itemgetter(2, 0, 1), reverse=True)
    B = branchings.greedy_branching(G, kind='max').edges(data=True)
    assert list(B) == [((2, 3), 5, {'weight': 1}), (3, 'a', {'weight': 1}), (2, 4, {'weight': 5})]