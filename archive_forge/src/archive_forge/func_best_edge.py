from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from math import isnan
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for, py_random_state
def best_edge(component):
    """Returns the optimum (minimum or maximum) edge on the edge
        boundary of the given set of nodes.

        A return value of ``None`` indicates an empty boundary.

        """
    sign = 1 if minimum else -1
    minwt = float('inf')
    boundary = None
    for e in nx.edge_boundary(G, component, data=True):
        wt = e[-1].get(weight, 1) * sign
        if isnan(wt):
            if ignore_nan:
                continue
            msg = f'NaN found as an edge weight. Edge {e}'
            raise ValueError(msg)
        if wt < minwt:
            minwt = wt
            boundary = e
    return boundary