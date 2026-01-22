from math import log
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(node_attrs='community')
def within_inter_cluster(G, ebunch=None, delta=0.001, community='community'):
    """Compute the ratio of within- and inter-cluster common neighbors
    of all node pairs in ebunch.

    For two nodes `u` and `v`, if a common neighbor `w` belongs to the
    same community as them, `w` is considered as within-cluster common
    neighbor of `u` and `v`. Otherwise, it is considered as
    inter-cluster common neighbor of `u` and `v`. The ratio between the
    size of the set of within- and inter-cluster common neighbors is
    defined as the WIC measure. [1]_

    Parameters
    ----------
    G : graph
        A NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        The WIC measure will be computed for each pair of nodes given in
        the iterable. The pairs must be given as 2-tuples (u, v) where
        u and v are nodes in the graph. If ebunch is None then all
        nonexistent edges in the graph will be used.
        Default value: None.

    delta : float, optional (default = 0.001)
        Value to prevent division by zero in case there is no
        inter-cluster common neighbor between two nodes. See [1]_ for
        details. Default value: 0.001.

    community : string, optional (default = 'community')
        Nodes attribute name containing the community information.
        G[u][community] identifies which community u belongs to. Each
        node belongs to at most one community. Default value: 'community'.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their WIC measure.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)])
    >>> G.nodes[0]["community"] = 0
    >>> G.nodes[1]["community"] = 1
    >>> G.nodes[2]["community"] = 0
    >>> G.nodes[3]["community"] = 0
    >>> G.nodes[4]["community"] = 0
    >>> preds = nx.within_inter_cluster(G, [(0, 4)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 4) -> 1.99800200
    >>> preds = nx.within_inter_cluster(G, [(0, 4)], delta=0.5)
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 4) -> 1.33333333

    References
    ----------
    .. [1] Jorge Carlos Valverde-Rebaza and Alneu de Andrade Lopes.
       Link prediction in complex networks based on cluster information.
       In Proceedings of the 21st Brazilian conference on Advances in
       Artificial Intelligence (SBIA'12)
       https://doi.org/10.1007/978-3-642-34459-6_10
    """
    if delta <= 0:
        raise nx.NetworkXAlgorithmError('Delta must be greater than zero')

    def predict(u, v):
        Cu = _community(G, u, community)
        Cv = _community(G, v, community)
        if Cu != Cv:
            return 0
        cnbors = set(nx.common_neighbors(G, u, v))
        within = {w for w in cnbors if _community(G, w, community) == Cu}
        inter = cnbors - within
        return len(within) / (len(inter) + delta)
    return _apply_prediction(G, predict, ebunch)