import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def bellman_ford_shortest_paths(graph, source, target=None, weight_fn=None, default_weight=1.0, as_undirected=False):
    """Find the shortest path from a node

    This function will generate the shortest path from a source node using
    the Bellman-Ford algorithm wit the SPFA heuristic.

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int source: The node index to find paths from
    :param int target: An optional target to find a path to
    :param weight_fn: An optional weight function for an edge. It will accept
        a single argument, the edge's weight object and will return a float
        which will be used to represent the weight/cost of the edge
    :param float default_weight: If ``weight_fn`` isn't specified this optional
        float value will be used for the weight/cost of each edge.
    :param bool as_undirected: If set to true the graph will be treated as
        undirected for finding the shortest path. This only works with a
        :class:`~rustworkx.PyDiGraph` input for ``graph``

    :return: A read-only dictionary of paths. The keys are destination node indices
        and the dict values are lists of node indices making the path.
    :rtype: PathMapping

    :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
        path is not defined
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))