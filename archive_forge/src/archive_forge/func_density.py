import networkx as nx
from networkx.algorithms.components import connected_components
from networkx.exception import AmbiguousSolution
@nx._dispatch(graphs='B')
def density(B, nodes):
    """Returns density of bipartite graph B.

    Parameters
    ----------
    B : NetworkX graph

    nodes: list or container
      Nodes in one node set of the bipartite graph.

    Returns
    -------
    d : float
       The bipartite density

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.complete_bipartite_graph(3, 2)
    >>> X = set([0, 1, 2])
    >>> bipartite.density(G, X)
    1.0
    >>> Y = set([3, 4])
    >>> bipartite.density(G, Y)
    1.0

    Notes
    -----
    The container of nodes passed as argument must contain all nodes
    in one of the two bipartite node sets to avoid ambiguity in the
    case of disconnected graphs.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    color
    """
    n = len(B)
    m = nx.number_of_edges(B)
    nb = len(nodes)
    nt = n - nb
    if m == 0:
        d = 0.0
    elif B.is_directed():
        d = m / (2 * nb * nt)
    else:
        d = m / (nb * nt)
    return d