from itertools import combinations
import networkx as nx
from networkx.algorithms.simple_paths import is_simple_path as is_path
from networkx.utils import arbitrary_element, not_implemented_for, py_random_state
@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch
def is_reachable(G, s, t):
    """Decides whether there is a path from `s` to `t` in the
    tournament.

    This function is more theoretically efficient than the reachability
    checks than the shortest path algorithms in
    :mod:`networkx.algorithms.shortest_paths`.

    The given graph **must** be a tournament, otherwise this function's
    behavior is undefined.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph representing a tournament.

    s : node
        A node in the graph.

    t : node
        A node in the graph.

    Returns
    -------
    bool
        Whether there is a path from `s` to `t` in `G`.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 0), (1, 3), (1, 2), (2, 3), (2, 0), (3, 0)])
    >>> nx.is_tournament(G)
    True
    >>> nx.tournament.is_reachable(G, 1, 3)
    True
    >>> nx.tournament.is_reachable(G, 3, 2)
    False

    Notes
    -----
    Although this function is more theoretically efficient than the
    generic shortest path functions, a speedup requires the use of
    parallelism. Though it may in the future, the current implementation
    does not use parallelism, thus you may not see much of a speedup.

    This algorithm comes from [1].

    References
    ----------
    .. [1] Tantau, Till.
           "A note on the complexity of the reachability problem for
           tournaments."
           *Electronic Colloquium on Computational Complexity*. 2001.
           <http://eccc.hpi-web.de/report/2001/092/>
    """

    def two_neighborhood(G, v):
        """Returns the set of nodes at distance at most two from `v`.

        `G` must be a graph and `v` a node in that graph.

        The returned set includes the nodes at distance zero (that is,
        the node `v` itself), the nodes at distance one (that is, the
        out-neighbors of `v`), and the nodes at distance two.

        """
        return {x for x in G if x == v or x in G[v] or any((is_path(G, [v, z, x]) for z in G))}

    def is_closed(G, nodes):
        """Decides whether the given set of nodes is closed.

        A set *S* of nodes is *closed* if for each node *u* in the graph
        not in *S* and for each node *v* in *S*, there is an edge from
        *u* to *v*.

        """
        return all((v in G[u] for u in set(G) - nodes for v in nodes))
    neighborhoods = [two_neighborhood(G, v) for v in G]
    return all((not (is_closed(G, S) and s in S and (t not in S)) for S in neighborhoods))