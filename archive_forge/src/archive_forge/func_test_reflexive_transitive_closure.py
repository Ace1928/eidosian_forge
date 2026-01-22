from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_reflexive_transitive_closure(self):
    G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    soln = sorted(solution + [(n, n) for n in G])
    assert edges_equal(nx.transitive_closure(G).edges(), solution)
    assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
    assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
    assert edges_equal(nx.transitive_closure(G, None).edges(), solution)
    G = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
    soln = sorted(solution + [(n, n) for n in G])
    assert edges_equal(nx.transitive_closure(G).edges(), solution)
    assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
    assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
    assert edges_equal(nx.transitive_closure(G, None).edges(), solution)
    G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = sorted([(1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1)])
    soln = sorted(solution + [(n, n) for n in G])
    assert edges_equal(sorted(nx.transitive_closure(G).edges()), soln)
    assert edges_equal(sorted(nx.transitive_closure(G, False).edges()), soln)
    assert edges_equal(sorted(nx.transitive_closure(G, None).edges()), solution)
    assert edges_equal(sorted(nx.transitive_closure(G, True).edges()), soln)
    G = nx.Graph([(1, 2), (2, 3), (3, 4)])
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    soln = sorted(solution + [(n, n) for n in G])
    assert edges_equal(nx.transitive_closure(G).edges(), solution)
    assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
    assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
    assert edges_equal(nx.transitive_closure(G, None).edges(), solution)
    G = nx.MultiGraph([(1, 2), (2, 3), (3, 4)])
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    soln = sorted(solution + [(n, n) for n in G])
    assert edges_equal(nx.transitive_closure(G).edges(), solution)
    assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
    assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
    assert edges_equal(nx.transitive_closure(G, None).edges(), solution)
    G = nx.MultiDiGraph([(1, 2), (2, 3), (3, 4)])
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    soln = sorted(solution + [(n, n) for n in G])
    assert edges_equal(nx.transitive_closure(G).edges(), solution)
    assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
    assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
    assert edges_equal(nx.transitive_closure(G, None).edges(), solution)