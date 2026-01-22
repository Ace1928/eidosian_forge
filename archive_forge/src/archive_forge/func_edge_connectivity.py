import itertools
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import (
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity
@nx._dispatch
def edge_connectivity(G, s=None, t=None, flow_func=None, cutoff=None):
    """Returns the edge connectivity of the graph or digraph G.

    The edge connectivity is equal to the minimum number of edges that
    must be removed to disconnect G or render it trivial. If source
    and target nodes are provided, this function returns the local edge
    connectivity: the minimum number of edges that must be removed to
    break all paths from source to target in G.

    Parameters
    ----------
    G : NetworkX graph
        Undirected or directed graph

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

    cutoff : integer, float, or None (default: None)
        If specified, the maximum flow algorithm will terminate when the
        flow value reaches or exceeds the cutoff. This only works for flows
        that support the cutoff parameter (most do) and is ignored otherwise.

    Returns
    -------
    K : integer
        Edge connectivity for G, or local edge connectivity if source
        and target were provided

    Examples
    --------
    >>> # Platonic icosahedral graph is 5-edge-connected
    >>> G = nx.icosahedral_graph()
    >>> nx.edge_connectivity(G)
    5

    You can use alternative flow algorithms for the underlying
    maximum flow computation. In dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better
    than the default :meth:`edmonds_karp`, which is faster for
    sparse networks with highly skewed degree distributions.
    Alternative flow functions have to be explicitly imported
    from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> nx.edge_connectivity(G, flow_func=shortest_augmenting_path)
    5

    If you specify a pair of nodes (source and target) as parameters,
    this function returns the value of local edge connectivity.

    >>> nx.edge_connectivity(G, 3, 7)
    5

    If you need to perform several local computations among different
    pairs of nodes on the same graph, it is recommended that you reuse
    the data structures used in the maximum flow computations. See
    :meth:`local_edge_connectivity` for details.

    Notes
    -----
    This is a flow based implementation of global edge connectivity.
    For undirected graphs the algorithm works by finding a 'small'
    dominating set of nodes of G (see algorithm 7 in [1]_ ) and
    computing local maximum flow (see :meth:`local_edge_connectivity`)
    between an arbitrary node in the dominating set and the rest of
    nodes in it. This is an implementation of algorithm 6 in [1]_ .
    For directed graphs, the algorithm does n calls to the maximum
    flow function. This is an implementation of algorithm 8 in [1]_ .

    See also
    --------
    :meth:`local_edge_connectivity`
    :meth:`local_node_connectivity`
    :meth:`node_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`
    :meth:`k_edge_components`
    :meth:`k_edge_subgraphs`

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
        return local_edge_connectivity(G, s, t, flow_func=flow_func, cutoff=cutoff)
    H = build_auxiliary_edge_connectivity(G)
    R = build_residual_network(H, 'capacity')
    kwargs = {'flow_func': flow_func, 'auxiliary': H, 'residual': R}
    if G.is_directed():
        if not nx.is_weakly_connected(G):
            return 0
        L = min((d for n, d in G.degree()))
        nodes = list(G)
        n = len(nodes)
        if cutoff is not None:
            L = min(cutoff, L)
        for i in range(n):
            kwargs['cutoff'] = L
            try:
                L = min(L, local_edge_connectivity(G, nodes[i], nodes[i + 1], **kwargs))
            except IndexError:
                L = min(L, local_edge_connectivity(G, nodes[i], nodes[0], **kwargs))
        return L
    else:
        if not nx.is_connected(G):
            return 0
        L = min((d for n, d in G.degree()))
        if cutoff is not None:
            L = min(cutoff, L)
        for node in G:
            D = nx.dominating_set(G, start_with=node)
            v = D.pop()
            if D:
                break
        else:
            return L
        for w in D:
            kwargs['cutoff'] = L
            L = min(L, local_edge_connectivity(G, v, w, **kwargs))
        return L