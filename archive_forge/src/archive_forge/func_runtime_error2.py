from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def runtime_error2():
    DG = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    first = True
    for x in algorithm(DG):
        if first:
            first = False
            DG.remove_node(2)