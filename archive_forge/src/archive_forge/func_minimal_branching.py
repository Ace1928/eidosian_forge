import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
@nx._dispatch(edge_attrs={'attr': 'default', 'partition': None}, preserve_edge_attrs='preserve_attrs')
def minimal_branching(G, /, *, attr='weight', default=1, preserve_attrs=False, partition=None):
    """
    Returns a minimal branching from `G`.

    A minimal branching is a branching similar to a minimal arborescence but
    without the requirement that the result is actually a spanning arborescence.
    This allows minimal branchinges to be computed over graphs which may not
    have arborescence (such as multiple components).

    Parameters
    ----------
    G : (multi)digraph-like
        The graph to be searched.
    attr : str
        The edge attribute used in determining optimality.
    default : float
        The value of the edge attribute used if an edge does not have
        the attribute `attr`.
    preserve_attrs : bool
        If True, preserve the other attributes of the original graph (that are not
        passed to `attr`)
    partition : str
        The key for the edge attribute containing the partition
        data on the graph. Edges can be included, excluded or open using the
        `EdgePartition` enum.

    Returns
    -------
    B : (multi)digraph-like
        A minimal branching.
    """
    max_weight = -INF
    min_weight = INF
    for _, _, w in G.edges(data=attr):
        if w > max_weight:
            max_weight = w
        if w < min_weight:
            min_weight = w
    for _, _, d in G.edges(data=True):
        d[attr] = max_weight + 1 + (max_weight - min_weight) - d[attr]
    B = maximum_branching(G, attr, default, preserve_attrs, partition)
    for _, _, d in G.edges(data=True):
        d[attr] = max_weight + 1 + (max_weight - min_weight) - d[attr]
    for _, _, d in B.edges(data=True):
        d[attr] = max_weight + 1 + (max_weight - min_weight) - d[attr]
    return B