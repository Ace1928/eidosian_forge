import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def vf2_mapping(first, second, node_matcher=None, edge_matcher=None, id_order=True, subgraph=False, induced=True, call_limit=None):
    """
    Return an iterator over all vf2 mappings between two graphs.

    This funcion will run the vf2 algorithm used from
    :func:`~rustworkx.is_isomorphic` and :func:`~rustworkx.is_subgraph_isomorphic`
    but instead of returning a boolean it will return an iterator over all possible
    mapping of node ids found from ``first`` to ``second``. If the graphs are not
    isomorphic then the iterator will be empty. A simple example that retrieves
    one mapping would be::

            graph_a = rustworkx.generators.path_graph(3)
            graph_b = rustworkx.generators.path_graph(2)
            vf2 = rustworkx.vf2_mapping(graph_a, graph_b, subgraph=True)
            try:
                mapping = next(vf2)
            except StopIteration:
                pass

    :param first: The first graph to find the mapping for
    :param second: The second graph to find the mapping for
    :param node_matcher: An optional python callable object that takes 2
        positional arguments, one for each node data object in either graph.
        If the return of this function evaluates to True then the nodes
        passed to it are viewed as matching.
    :param edge_matcher: A python callable object that takes 2 positional
        one for each edge data object. If the return of this
        function evaluates to True then the edges passed to it are viewed
        as matching.
    :param bool id_order: If set to ``False`` this function will use a
        heuristic matching order based on [VF2]_ paper. Otherwise it will
        default to matching the nodes in order specified by their ids.
    :param bool subgraph: If set to ``True`` the function will return the
        subgraph isomorphic found between the graphs.
    :param bool induced: If set to ``True`` this function will check the existence
        of a node-induced subgraph of first isomorphic to second graph.
        Default: ``True``.
    :param int call_limit: An optional bound on the number of states that VF2
        algorithm visits while searching for a solution. If it exceeds this limit,
        the algorithm will stop. Default: ``None``.

    :returns: An iterator over dicitonaries of node indices from ``first`` to node
        indices in ``second`` representing the mapping found.
    :rtype: Iterable[NodeMap]
    """
    raise TypeError('Invalid Input Type %s for graph' % type(first))