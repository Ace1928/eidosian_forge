import itertools
import networkx as nx
@nx._dispatch
def latapy_clustering(G, nodes=None, mode='dot'):
    """Compute a bipartite clustering coefficient for nodes.

    The bipartite clustering coefficient is a measure of local density
    of connections defined as [1]_:

    .. math::

       c_u = \\frac{\\sum_{v \\in N(N(u))} c_{uv} }{|N(N(u))|}

    where `N(N(u))` are the second order neighbors of `u` in `G` excluding `u`,
    and `c_{uv}` is the pairwise clustering coefficient between nodes
    `u` and `v`.

    The mode selects the function for `c_{uv}` which can be:

    `dot`:

    .. math::

       c_{uv}=\\frac{|N(u)\\cap N(v)|}{|N(u) \\cup N(v)|}

    `min`:

    .. math::

       c_{uv}=\\frac{|N(u)\\cap N(v)|}{min(|N(u)|,|N(v)|)}

    `max`:

    .. math::

       c_{uv}=\\frac{|N(u)\\cap N(v)|}{max(|N(u)|,|N(v)|)}


    Parameters
    ----------
    G : graph
        A bipartite graph

    nodes : list or iterable (optional)
        Compute bipartite clustering for these nodes. The default
        is all nodes in G.

    mode : string
        The pairwise bipartite clustering method to be used in the computation.
        It must be "dot", "max", or "min".

    Returns
    -------
    clustering : dictionary
        A dictionary keyed by node with the clustering coefficient value.


    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)  # path graphs are bipartite
    >>> c = bipartite.clustering(G)
    >>> c[0]
    0.5
    >>> c = bipartite.clustering(G, mode="min")
    >>> c[0]
    1.0

    See Also
    --------
    robins_alexander_clustering
    average_clustering
    networkx.algorithms.cluster.square_clustering

    References
    ----------
    .. [1] Latapy, Matthieu, Clémence Magnien, and Nathalie Del Vecchio (2008).
       Basic notions for the analysis of large two-mode networks.
       Social Networks 30(1), 31--48.
    """
    if not nx.algorithms.bipartite.is_bipartite(G):
        raise nx.NetworkXError('Graph is not bipartite')
    try:
        cc_func = modes[mode]
    except KeyError as err:
        raise nx.NetworkXError('Mode for bipartite clustering must be: dot, min or max') from err
    if nodes is None:
        nodes = G
    ccs = {}
    for v in nodes:
        cc = 0.0
        nbrs2 = {u for nbr in G[v] for u in G[nbr]} - {v}
        for u in nbrs2:
            cc += cc_func(set(G[u]), set(G[v]))
        if cc > 0.0:
            cc /= len(nbrs2)
        ccs[v] = cc
    return ccs