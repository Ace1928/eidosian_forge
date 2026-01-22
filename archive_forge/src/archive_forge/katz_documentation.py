import math
import networkx as nx
from networkx.utils import not_implemented_for
Compute the Katz centrality for the graph G.

    Katz centrality computes the centrality for a node based on the centrality
    of its neighbors. It is a generalization of the eigenvector centrality. The
    Katz centrality for node $i$ is

    .. math::

        x_i = \alpha \sum_{j} A_{ij} x_j + \beta,

    where $A$ is the adjacency matrix of graph G with eigenvalues $\lambda$.

    The parameter $\beta$ controls the initial centrality and

    .. math::

        \alpha < \frac{1}{\lambda_{\max}}.

    Katz centrality computes the relative influence of a node within a
    network by measuring the number of the immediate neighbors (first
    degree nodes) and also all other nodes in the network that connect
    to the node under consideration through these immediate neighbors.

    Extra weight can be provided to immediate neighbors through the
    parameter $\beta$.  Connections made with distant neighbors
    are, however, penalized by an attenuation factor $\alpha$ which
    should be strictly less than the inverse largest eigenvalue of the
    adjacency matrix in order for the Katz centrality to be computed
    correctly. More information is provided in [1]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    alpha : float
      Attenuation factor

    beta : scalar or dictionary, optional (default=1.0)
      Weight attributed to the immediate neighborhood. If not a scalar the
      dictionary must have an value for every node.

    normalized : bool
      If True normalize the resulting values.

    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      In this measure the weight is interpreted as the connection strength.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with Katz centrality as the value.

    Raises
    ------
    NetworkXError
       If the parameter `beta` is not a scalar but lacks a value for at least
       one node

    Examples
    --------
    >>> import math
    >>> G = nx.path_graph(4)
    >>> phi = (1 + math.sqrt(5)) / 2.0  # largest eigenvalue of adj matrix
    >>> centrality = nx.katz_centrality_numpy(G, 1 / phi)
    >>> for n, c in sorted(centrality.items()):
    ...     print(f"{n} {c:.2f}")
    0 0.37
    1 0.60
    2 0.60
    3 0.37

    See Also
    --------
    katz_centrality
    eigenvector_centrality_numpy
    eigenvector_centrality
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Katz centrality was introduced by [2]_.

    This algorithm uses a direct linear solver to solve the above equation.
    The parameter ``alpha`` should be strictly less than the inverse of largest
    eigenvalue of the adjacency matrix for there to be a solution.
    You can use ``max(nx.adjacency_spectrum(G))`` to get $\lambda_{\max}$ the largest
    eigenvalue of the adjacency matrix.

    When $\alpha = 1/\lambda_{\max}$ and $\beta=0$, Katz centrality is the same
    as eigenvector centrality.

    For directed graphs this finds "left" eigenvectors which corresponds
    to the in-edges in the graph. For out-edges Katz centrality
    first reverse the graph with ``G.reverse()``.

    References
    ----------
    .. [1] Mark E. J. Newman:
       Networks: An Introduction.
       Oxford University Press, USA, 2010, p. 173.
    .. [2] Leo Katz:
       A New Status Index Derived from Sociometric Index.
       Psychometrika 18(1):39–43, 1953
       https://link.springer.com/content/pdf/10.1007/BF02289026.pdf
    