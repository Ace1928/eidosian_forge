import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
@nx._dispatch(preserve_all_attrs=True)
def k_corona(G, k, core_number=None):
    """Returns the k-corona of G.

    The k-corona is the subgraph of nodes in the k-core which have
    exactly k neighbours in the k-core.

    Parameters
    ----------
    G : NetworkX graph
       A graph or directed graph
    k : int
       The order of the corona.
    core_number : dictionary, optional
       Precomputed core numbers for the graph G.

    Returns
    -------
    G : NetworkX graph
       The k-corona subgraph

    Raises
    ------
    NetworkXError
        The k-corona is not defined for graphs with self loops or
        parallel edges.

    Notes
    -----
    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Graph, node, and edge attributes are copied to the subgraph.

    See Also
    --------
    core_number

    References
    ----------
    .. [1]  k -core (bootstrap) percolation on complex networks:
       Critical phenomena and nonlocal effects,
       A. V. Goltsev, S. N. Dorogovtsev, and J. F. F. Mendes,
       Phys. Rev. E 73, 056101 (2006)
       http://link.aps.org/doi/10.1103/PhysRevE.73.056101
    """

    def func(v, k, c):
        return c[v] == k and k == sum((1 for w in G[v] if c[w] >= k))
    return _core_subgraph(G, func, k, core_number)