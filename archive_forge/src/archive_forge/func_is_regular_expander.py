import itertools
import networkx as nx
@nx.utils.not_implemented_for('directed')
@nx.utils.not_implemented_for('multigraph')
@nx._dispatchable(preserve_edge_attrs={'G': {'weight': 1}})
def is_regular_expander(G, *, epsilon=0):
    """Determines whether the graph G is a regular expander. [1]_

    An expander graph is a sparse graph with strong connectivity properties.

    More precisely, this helper checks whether the graph is a
    regular $(n, d, \\lambda)$-expander with $\\lambda$ close to
    the Alon-Boppana bound and given by
    $\\lambda = 2 \\sqrt{d - 1} + \\epsilon$. [2]_

    In the case where $\\epsilon = 0$ then if the graph successfully passes the test
    it is a Ramanujan graph. [3]_

    A Ramanujan graph has spectral gap almost as large as possible, which makes them
    excellent expanders.

    Parameters
    ----------
    G : NetworkX graph
    epsilon : int, float, default=0

    Returns
    -------
    bool
        Whether the given graph is a regular $(n, d, \\lambda)$-expander
        where $\\lambda = 2 \\sqrt{d - 1} + \\epsilon$.

    Examples
    --------
    >>> G = nx.random_regular_expander_graph(20, 4)
    >>> nx.is_regular_expander(G)
    True

    See Also
    --------
    maybe_regular_expander
    random_regular_expander_graph

    References
    ----------
    .. [1] Expander graph, https://en.wikipedia.org/wiki/Expander_graph
    .. [2] Alon-Boppana bound, https://en.wikipedia.org/wiki/Alon%E2%80%93Boppana_bound
    .. [3] Ramanujan graphs, https://en.wikipedia.org/wiki/Ramanujan_graph

    """
    import numpy as np
    from scipy.sparse.linalg import eigsh
    if epsilon < 0:
        raise nx.NetworkXError('epsilon must be non negative')
    if not nx.is_regular(G):
        return False
    _, d = nx.utils.arbitrary_element(G.degree)
    A = nx.adjacency_matrix(G, dtype=float)
    lams = eigsh(A, which='LM', k=2, return_eigenvectors=False)
    lambda2 = min(lams)
    return bool(abs(lambda2) < 2 ** np.sqrt(d - 1) + epsilon)