import itertools
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import (
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity
@nx._dispatch
def average_node_connectivity(G, flow_func=None):
    """Returns the average connectivity of a graph G.

    The average connectivity `\\bar{\\kappa}` of a graph G is the average
    of local node connectivity over all pairs of nodes of G [1]_ .

    .. math::

        \\bar{\\kappa}(G) = \\frac{\\sum_{u,v} \\kappa_{G}(u,v)}{{n \\choose 2}}

    Parameters
    ----------

    G : NetworkX graph
        Undirected graph

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See :meth:`local_node_connectivity`
        for details. The choice of the default function may change from
        version to version and should not be relied on. Default value: None.

    Returns
    -------
    K : float
        Average node connectivity

    See also
    --------
    :meth:`local_node_connectivity`
    :meth:`node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1]  Beineke, L., O. Oellermann, and R. Pippert (2002). The average
            connectivity of a graph. Discrete mathematics 252(1-3), 31-45.
            http://www.sciencedirect.com/science/article/pii/S0012365X01001807

    """
    if G.is_directed():
        iter_func = itertools.permutations
    else:
        iter_func = itertools.combinations
    H = build_auxiliary_node_connectivity(G)
    R = build_residual_network(H, 'capacity')
    kwargs = {'flow_func': flow_func, 'auxiliary': H, 'residual': R}
    num, den = (0, 0)
    for u, v in iter_func(G, 2):
        num += local_node_connectivity(G, u, v, **kwargs)
        den += 1
    if den == 0:
        return 0
    return num / den