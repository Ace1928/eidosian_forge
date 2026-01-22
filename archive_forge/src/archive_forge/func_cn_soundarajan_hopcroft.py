from math import log
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(node_attrs='community')
def cn_soundarajan_hopcroft(G, ebunch=None, community='community'):
    """Count the number of common neighbors of all node pairs in ebunch
        using community information.

    For two nodes $u$ and $v$, this function computes the number of
    common neighbors and bonus one for each common neighbor belonging to
    the same community as $u$ and $v$. Mathematically,

    .. math::

        |\\Gamma(u) \\cap \\Gamma(v)| + \\sum_{w \\in \\Gamma(u) \\cap \\Gamma(v)} f(w)

    where $f(w)$ equals 1 if $w$ belongs to the same community as $u$
    and $v$ or 0 otherwise and $\\Gamma(u)$ denotes the set of
    neighbors of $u$.

    Parameters
    ----------
    G : graph
        A NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        The score will be computed for each pair of nodes given in the
        iterable. The pairs must be given as 2-tuples (u, v) where u
        and v are nodes in the graph. If ebunch is None then all
        nonexistent edges in the graph will be used.
        Default value: None.

    community : string, optional (default = 'community')
        Nodes attribute name containing the community information.
        G[u][community] identifies which community u belongs to. Each
        node belongs to at most one community. Default value: 'community'.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their score.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> G.nodes[0]["community"] = 0
    >>> G.nodes[1]["community"] = 0
    >>> G.nodes[2]["community"] = 0
    >>> preds = nx.cn_soundarajan_hopcroft(G, [(0, 2)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p}")
    (0, 2) -> 2

    References
    ----------
    .. [1] Sucheta Soundarajan and John Hopcroft.
       Using community information to improve the precision of link
       prediction methods.
       In Proceedings of the 21st international conference companion on
       World Wide Web (WWW '12 Companion). ACM, New York, NY, USA, 607-608.
       http://doi.acm.org/10.1145/2187980.2188150
    """

    def predict(u, v):
        Cu = _community(G, u, community)
        Cv = _community(G, v, community)
        cnbors = list(nx.common_neighbors(G, u, v))
        neighbors = sum((_community(G, w, community) == Cu for w in cnbors)) if Cu == Cv else 0
        return len(cnbors) + neighbors
    return _apply_prediction(G, predict, ebunch)