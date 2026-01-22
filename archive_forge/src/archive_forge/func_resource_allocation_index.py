from math import log
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def resource_allocation_index(G, ebunch=None):
    """Compute the resource allocation index of all node pairs in ebunch.

    Resource allocation index of `u` and `v` is defined as

    .. math::

        \\sum_{w \\in \\Gamma(u) \\cap \\Gamma(v)} \\frac{1}{|\\Gamma(w)|}

    where $\\Gamma(u)$ denotes the set of neighbors of $u$.

    Parameters
    ----------
    G : graph
        A NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Resource allocation index will be computed for each pair of
        nodes given in the iterable. The pairs must be given as
        2-tuples (u, v) where u and v are nodes in the graph. If ebunch
        is None then all nonexistent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their resource allocation index.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = nx.resource_allocation_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.75000000
    (2, 3) -> 0.75000000

    References
    ----------
    .. [1] T. Zhou, L. Lu, Y.-C. Zhang.
       Predicting missing links via local information.
       Eur. Phys. J. B 71 (2009) 623.
       https://arxiv.org/pdf/0901.0553.pdf
    """

    def predict(u, v):
        return sum((1 / G.degree(w) for w in nx.common_neighbors(G, u, v)))
    return _apply_prediction(G, predict, ebunch)