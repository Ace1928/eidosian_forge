from collections import defaultdict, deque
from itertools import chain, combinations, islice
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch
def find_cliques_recursive(G, nodes=None):
    """Returns all maximal cliques in a graph.

    For each node *v*, a *maximal clique for v* is a largest complete
    subgraph containing *v*. The largest maximal clique is sometimes
    called the *maximum clique*.

    This function returns an iterator over cliques, each of which is a
    list of nodes. It is a recursive implementation, so may suffer from
    recursion depth issues, but is included for pedagogical reasons.
    For a non-recursive implementation, see :func:`find_cliques`.

    This function accepts a list of `nodes` and only the maximal cliques
    containing all of these `nodes` are returned. It can considerably speed up
    the running time if some specific cliques are desired.

    Parameters
    ----------
    G : NetworkX graph

    nodes : list, optional (default=None)
        If provided, only yield *maximal cliques* containing all nodes in `nodes`.
        If `nodes` isn't a clique itself, a ValueError is raised.

    Returns
    -------
    iterator
        An iterator over maximal cliques, each of which is a list of
        nodes in `G`. If `nodes` is provided, only the maximal cliques
        containing all the nodes in `nodes` are yielded. The order of
        cliques is arbitrary.

    Raises
    ------
    ValueError
        If `nodes` is not a clique.

    See Also
    --------
    find_cliques
        An iterative version of the same algorithm. See docstring for examples.

    Notes
    -----
    To obtain a list of all maximal cliques, use
    `list(find_cliques_recursive(G))`. However, be aware that in the
    worst-case, the length of this list can be exponential in the number
    of nodes in the graph. This function avoids storing all cliques in memory
    by only keeping current candidate node lists in memory during its search.

    This implementation is based on the algorithm published by Bron and
    Kerbosch (1973) [1]_, as adapted by Tomita, Tanaka and Takahashi
    (2006) [2]_ and discussed in Cazals and Karande (2008) [3]_. For a
    non-recursive implementation, see :func:`find_cliques`.

    This algorithm ignores self-loops and parallel edges, since cliques
    are not conventionally defined with such edges.

    References
    ----------
    .. [1] Bron, C. and Kerbosch, J.
       "Algorithm 457: finding all cliques of an undirected graph".
       *Communications of the ACM* 16, 9 (Sep. 1973), 575--577.
       <http://portal.acm.org/citation.cfm?doid=362342.362367>

    .. [2] Etsuji Tomita, Akira Tanaka, Haruhisa Takahashi,
       "The worst-case time complexity for generating all maximal
       cliques and computational experiments",
       *Theoretical Computer Science*, Volume 363, Issue 1,
       Computing and Combinatorics,
       10th Annual International Conference on
       Computing and Combinatorics (COCOON 2004), 25 October 2006, Pages 28--42
       <https://doi.org/10.1016/j.tcs.2006.06.015>

    .. [3] F. Cazals, C. Karande,
       "A note on the problem of reporting maximal cliques",
       *Theoretical Computer Science*,
       Volume 407, Issues 1--3, 6 November 2008, Pages 564--568,
       <https://doi.org/10.1016/j.tcs.2008.05.010>

    """
    if len(G) == 0:
        return iter([])
    adj = {u: {v for v in G[u] if v != u} for u in G}
    Q = nodes[:] if nodes is not None else []
    cand_init = set(G)
    for node in Q:
        if node not in cand_init:
            raise ValueError(f'The given `nodes` {nodes} do not form a clique')
        cand_init &= adj[node]
    if not cand_init:
        return iter([Q])
    subg_init = cand_init.copy()

    def expand(subg, cand):
        u = max(subg, key=lambda u: len(cand & adj[u]))
        for q in cand - adj[u]:
            cand.remove(q)
            Q.append(q)
            adj_q = adj[q]
            subg_q = subg & adj_q
            if not subg_q:
                yield Q[:]
            else:
                cand_q = cand & adj_q
                if cand_q:
                    yield from expand(subg_q, cand_q)
            Q.pop()
    return expand(subg_init, cand_init)