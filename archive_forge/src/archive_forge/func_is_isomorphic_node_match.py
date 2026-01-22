import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def is_isomorphic_node_match(first, second, matcher, id_order=True):
    """Determine if 2 graphs are isomorphic

    This checks if 2 graphs are isomorphic both structurally and also
    comparing the node data using the provided matcher function. The matcher
    function takes in 2 node data objects and will compare them. A simple
    example that checks if they're just equal would be::

        graph_a = rustworkx.PyDAG()
        graph_b = rustworkx.PyDAG()
        rustworkx.is_isomorphic_node_match(graph_a, graph_b,
                                        lambda x, y: x == y)

    .. note::

        For better performance on large graphs, consider setting
        `id_order=False`.

    :param first: The first graph to compare. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param second: The second graph to compare. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
        It should be the same type as the first graph.
    :param callable matcher: A python callable object that takes 2 positional
        one for each node data object. If the return of this
        function evaluates to True then the nodes passed to it are vieded
        as matching.
    :param bool id_order: If set to ``False`` this function will use a
        heuristic matching order based on [VF2]_ paper. Otherwise it will
        default to matching the nodes in order specified by their ids.

    :returns: ``True`` if the 2 graphs are isomorphic ``False`` if they are
        not.
    :rtype: bool
    """
    raise TypeError('Invalid Input Type %s for graph' % type(first))