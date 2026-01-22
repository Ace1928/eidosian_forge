import itertools
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import (
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity
@nx._dispatch
def node_connectivity(G, s=None, t=None, flow_func=None):
    """Returns node connectivity for a graph or digraph G.

    Node connectivity is equal to the minimum number of nodes that
    must be removed to disconnect G or render it trivial. If source
    and target nodes are provided, this function returns the local node
    connectivity: the minimum number of nodes that must be removed to break
    all paths from source to target in G.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    s : node
        Source node. Optional. Default value: None.

    t : node
        Target node. Optional. Default value: None.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The
        choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    Returns
    -------
    K : integer
        Node connectivity of G, or local node connectivity if source
        and target are provided.

    Examples
    --------
    >>> # Platonic icosahedral graph is 5-node-connected
    >>> G = nx.icosahedral_graph()
    >>> nx.node_connectivity(G)
    5

    You can use alternative flow algorithms for the underlying maximum
    flow computation. In dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better
    than the default :meth:`edmonds_karp`, which is faster for
    sparse networks with highly skewed degree distributions. Alternative
    flow functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> nx.node_connectivity(G, flow_func=shortest_augmenting_path)
    5

    If you specify a pair of nodes (source and target) as parameters,
    this function returns the value of local node connectivity.

    >>> nx.node_connectivity(G, 3, 7)
    5

    If you need to perform several local computations among different
    pairs of nodes on the same graph, it is recommended that you reuse
    the data structures used in the maximum flow computations. See
    :meth:`local_node_connectivity` for details.

    Notes
    -----
    This is a flow based implementation of node connectivity. The
    algorithm works by solving $O((n-\\delta-1+\\delta(\\delta-1)/2))$
    maximum flow problems on an auxiliary digraph. Where $\\delta$
    is the minimum degree of G. For details about the auxiliary
    digraph and the computation of local node connectivity see
    :meth:`local_node_connectivity`. This implementation is based
    on algorithm 11 in [1]_.

    See also
    --------
    :meth:`local_node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    if s is not None and t is None or (s is None and t is not None):
        raise nx.NetworkXError('Both source and target must be specified.')
    if s is not None and t is not None:
        if s not in G:
            raise nx.NetworkXError(f'node {s} not in graph')
        if t not in G:
            raise nx.NetworkXError(f'node {t} not in graph')
        return local_node_connectivity(G, s, t, flow_func=flow_func)
    if G.is_directed():
        if not nx.is_weakly_connected(G):
            return 0
        iter_func = itertools.permutations

        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors(v), G.successors(v)])
    else:
        if not nx.is_connected(G):
            return 0
        iter_func = itertools.combinations
        neighbors = G.neighbors
    H = build_auxiliary_node_connectivity(G)
    R = build_residual_network(H, 'capacity')
    kwargs = {'flow_func': flow_func, 'auxiliary': H, 'residual': R}
    v, K = min(G.degree(), key=itemgetter(1))
    for w in set(G) - set(neighbors(v)) - {v}:
        kwargs['cutoff'] = K
        K = min(K, local_node_connectivity(G, v, w, **kwargs))
    for x, y in iter_func(neighbors(v), 2):
        if y in G[x]:
            continue
        kwargs['cutoff'] = K
        K = min(K, local_node_connectivity(G, x, y, **kwargs))
    return K