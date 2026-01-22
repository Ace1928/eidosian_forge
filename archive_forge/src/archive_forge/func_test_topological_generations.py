from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_topological_generations():
    G = nx.DiGraph({1: [2, 3], 2: [4, 5], 3: [7], 4: [], 5: [6, 7], 6: [], 7: []}).reverse()
    generations = [sorted(gen) for gen in nx.topological_generations(G)]
    expected = [[4, 6, 7], [3, 5], [2], [1]]
    assert generations == expected
    MG = nx.MultiDiGraph(G.edges)
    MG.add_edge(2, 1)
    generations = [sorted(gen) for gen in nx.topological_generations(MG)]
    assert generations == expected