import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
@nx._dispatch(edge_attrs={'attr': 'default', 'partition': None}, preserve_edge_attrs='preserve_attrs')
def minimum_branching(G, attr='weight', default=1, preserve_attrs=False, partition=None):
    for _, _, d in G.edges(data=True):
        d[attr] = -d[attr]
    B = maximum_branching(G, attr, default, preserve_attrs, partition)
    for _, _, d in G.edges(data=True):
        d[attr] = -d[attr]
    for _, _, d in B.edges(data=True):
        d[attr] = -d[attr]
    return B