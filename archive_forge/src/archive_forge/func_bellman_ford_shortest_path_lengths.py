import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def bellman_ford_shortest_path_lengths(graph, node, edge_cost_fn, goal=None):
    """Compute the lengths of the shortest paths for a graph object using
    the Bellman-Ford algorithm with the SPFA heuristic.

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int node: The node index to use as the source for finding the
        shortest paths from
    :param edge_cost_fn: A python callable that will take in 1 parameter, an
        edge's data object and will return a float that represents the
        cost/weight of that edge. It can be negative.
    :param int goal: An optional node index to use as the end of the path.
        When specified the output dictionary will only have a single entry with
        the length of the shortest path to the goal node.

    :returns: A read-only dictionary of the shortest paths from the provided node
        where the key is the node index of the end of the path and the value is the
        cost/sum of the weights of path
    :rtype: PathLengthMapping

    :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
        path is not defined
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))