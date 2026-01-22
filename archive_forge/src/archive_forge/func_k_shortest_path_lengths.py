import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def k_shortest_path_lengths(graph, start, k, edge_cost, goal=None):
    """Compute the length of the kth shortest path

    Computes the lengths of the kth shortest path from ``start`` to every
    reachable node.

    Computes in :math:`O(k * (|E| + |V|*log(|V|)))` time (average).

    :param graph: The graph to find the shortest paths in. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int start: The node index to find the shortest paths from
    :param int k: The kth shortest path to find the lengths of
    :param edge_cost: A python callable that will receive an edge payload and
        return a float for the cost of that eedge
    :param int goal: An optional goal node index, if specified the output
        dictionary

    :returns: A dict of lengths where the key is the destination node index and
        the value is the length of the path.
    :rtype: dict
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))