import networkx as nx
from networkx.exception import NetworkXError
@nx._dispatch(graphs=None)
def hnm_harary_graph(n, m, create_using=None):
    """Returns the Harary graph with given numbers of nodes and edges.

    The Harary graph $H_{n,m}$ is the graph that maximizes node connectivity
    with $n$ nodes and $m$ edges.

    This maximum node connectivity is known to be floor($2m/n$). [1]_

    Parameters
    ----------
    n: integer
       The number of nodes the generated graph is to contain

    m: integer
       The number of edges the generated graph is to contain

    create_using : NetworkX graph constructor, optional Graph type
     to create (default=nx.Graph). If graph instance, then cleared
     before populated.

    Returns
    -------
    NetworkX graph
        The Harary graph $H_{n,m}$.

    See Also
    --------
    hkn_harary_graph

    Notes
    -----
    This algorithm runs in $O(m)$ time.
    It is implemented by following the Reference [2]_.

    References
    ----------
    .. [1] F. T. Boesch, A. Satyanarayana, and C. L. Suffel,
       "A Survey of Some Network Reliability Analysis and Synthesis Results,"
       Networks, pp. 99-107, 2009.

    .. [2] Harary, F. "The Maximum Connectivity of a Graph."
       Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.
    """
    if n < 1:
        raise NetworkXError('The number of nodes must be >= 1!')
    if m < n - 1:
        raise NetworkXError('The number of edges must be >= n - 1 !')
    if m > n * (n - 1) // 2:
        raise NetworkXError('The number of edges must be <= n(n-1)/2')
    H = nx.empty_graph(n, create_using)
    d = 2 * m // n
    if n % 2 == 0 or d % 2 == 0:
        offset = d // 2
        for i in range(n):
            for j in range(1, offset + 1):
                H.add_edge(i, (i - j) % n)
                H.add_edge(i, (i + j) % n)
        if d & 1:
            half = n // 2
            for i in range(half):
                H.add_edge(i, i + half)
        r = 2 * m % n
        if r > 0:
            for i in range(r // 2):
                H.add_edge(i, i + offset + 1)
    else:
        offset = (d - 1) // 2
        for i in range(n):
            for j in range(1, offset + 1):
                H.add_edge(i, (i - j) % n)
                H.add_edge(i, (i + j) % n)
        half = n // 2
        for i in range(m - n * offset):
            H.add_edge(i, (i + half) % n)
    return H