import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
@nx._dispatch(edge_attrs={'attr': 'default', 'partition': None}, preserve_edge_attrs='preserve_attrs')
def maximum_spanning_arborescence(G, attr='weight', default=1, preserve_attrs=False, partition=None):
    min_weight = INF
    max_weight = -INF
    for _, _, w in G.edges(data=attr):
        if w < min_weight:
            min_weight = w
        if w > max_weight:
            max_weight = w
    for _, _, d in G.edges(data=True):
        d[attr] = d[attr] - min_weight + 1 - (min_weight - max_weight)
    B = maximum_branching(G, attr, default, preserve_attrs, partition)
    for _, _, d in G.edges(data=True):
        d[attr] = d[attr] + min_weight - 1 + (min_weight - max_weight)
    for _, _, d in B.edges(data=True):
        d[attr] = d[attr] + min_weight - 1 + (min_weight - max_weight)
    if not is_arborescence(B):
        raise nx.exception.NetworkXException('No maximum spanning arborescence in G.')
    return B