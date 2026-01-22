import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
@nx._dispatch
def has_cycle(G):
    """Decides whether the directed graph has a cycle."""
    try:
        deque(topological_sort(G), maxlen=0)
    except nx.NetworkXUnfeasible:
        return True
    else:
        return False