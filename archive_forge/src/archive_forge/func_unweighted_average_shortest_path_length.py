import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def unweighted_average_shortest_path_length(graph, parallel_threshold=300, disconnected=False):
    """Return the average shortest path length with unweighted edges.

    The average shortest path length is calculated as

    .. math::

        a =\\sum_{s,t \\in V, s \\ne t} \\frac{d(s, t)}{n(n-1)}

    where :math:`V` is the set of nodes in ``graph``, :math:`d(s, t)` is the
    shortest path length from :math:`s` to :math:`t`, and :math:`n` is the
    number of nodes in ``graph``. If ``disconnected`` is set to ``True``,
    the average will be taken only between connected nodes.

    This function is also multithreaded and will run in parallel if the number
    of nodes in the graph is above the value of ``parallel_threshold`` (it
    defaults to 300). If the function will be running in parallel the env var
    ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
    By default it will use all available CPUs if the environment variable is
    not specified.

    :param graph: The graph to compute the average shortest path length for,
        can be either a :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param int parallel_threshold: The number of nodes to calculate the
        the distance matrix in parallel at. It defaults to 300, but this can
        be tuned to any number of nodes.
    :param bool as_undirected: If set to ``True`` the input directed graph
        will be treated as if each edge was bidirectional/undirected while
        finding the shortest paths. Default: ``False``.
    :param bool disconnected: If set to ``True`` only connected vertex pairs
        will be included in the calculation. If ``False``, infinity is returned
        for disconnected graphs. Default: ``False``.

    :returns: The average shortest path length. If no vertex pairs can be included
        in the calculation this will return NaN.

    :rtype: float
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))