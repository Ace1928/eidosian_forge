from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def worst_case_graph(self, k):
    G = nx.DiGraph()
    for n in range(2, k + 2):
        G.add_edge(1, n)
        G.add_edge(n, k + 2)
    G.add_edge(2 * k + 1, 1)
    for n in range(k + 2, 2 * k + 2):
        G.add_edge(n, 2 * k + 2)
        G.add_edge(n, n + 1)
    G.add_edge(2 * k + 3, k + 2)
    for n in range(2 * k + 3, 3 * k + 3):
        G.add_edge(2 * k + 2, n)
        G.add_edge(n, 3 * k + 3)
    G.add_edge(3 * k + 3, 2 * k + 2)
    return G