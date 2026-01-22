from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
@nx._dispatch
def simple_cycles(G, length_bound=None):
    """Find simple cycles (elementary circuits) of a graph.

    A `simple cycle`, or `elementary circuit`, is a closed path where
    no node appears twice.  In a directed graph, two simple cycles are distinct
    if they are not cyclic permutations of each other.  In an undirected graph,
    two simple cycles are distinct if they are not cyclic permutations of each
    other nor of the other's reversal.

    Optionally, the cycles are bounded in length.  In the unbounded case, we use
    a nonrecursive, iterator/generator version of Johnson's algorithm [1]_.  In
    the bounded case, we use a version of the algorithm of Gupta and
    Suzumura[2]_. There may be better algorithms for some cases [3]_ [4]_ [5]_.

    The algorithms of Johnson, and Gupta and Suzumura, are enhanced by some
    well-known preprocessing techniques.  When G is directed, we restrict our
    attention to strongly connected components of G, generate all simple cycles
    containing a certain node, remove that node, and further decompose the
    remainder into strongly connected components.  When G is undirected, we
    restrict our attention to biconnected components, generate all simple cycles
    containing a particular edge, remove that edge, and further decompose the
    remainder into biconnected components.

    Note that multigraphs are supported by this function -- and in undirected
    multigraphs, a pair of parallel edges is considered a cycle of length 2.
    Likewise, self-loops are considered to be cycles of length 1.  We define
    cycles as sequences of nodes; so the presence of loops and parallel edges
    does not change the number of simple cycles in a graph.

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph

    length_bound : int or None, optional (default=None)
       If length_bound is an int, generate all simple cycles of G with length at
       most length_bound.  Otherwise, generate all simple cycles of G.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    Examples
    --------
    >>> edges = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    >>> G = nx.DiGraph(edges)
    >>> sorted(nx.simple_cycles(G))
    [[0], [0, 1, 2], [0, 2], [1, 2], [2]]

    To filter the cycles so that they don't include certain nodes or edges,
    copy your graph and eliminate those nodes or edges before calling.
    For example, to exclude self-loops from the above example:

    >>> H = G.copy()
    >>> H.remove_edges_from(nx.selfloop_edges(G))
    >>> sorted(nx.simple_cycles(H))
    [[0, 1, 2], [0, 2], [1, 2]]

    Notes
    -----
    When length_bound is None, the time complexity is $O((n+e)(c+1))$ for $n$
    nodes, $e$ edges and $c$ simple circuits.  Otherwise, when length_bound > 1,
    the time complexity is $O((c+n)(k-1)d^k)$ where $d$ is the average degree of
    the nodes of G and $k$ = length_bound.

    Raises
    ------
    ValueError
        when length_bound < 0.

    References
    ----------
    .. [1] Finding all the elementary circuits of a directed graph.
       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.
       https://doi.org/10.1137/0204007
    .. [2] Finding All Bounded-Length Simple Cycles in a Directed Graph
       A. Gupta and T. Suzumura https://arxiv.org/abs/2105.10094
    .. [3] Enumerating the cycles of a digraph: a new preprocessing strategy.
       G. Loizou and P. Thanish, Information Sciences, v. 27, 163-182, 1982.
    .. [4] A search strategy for the elementary cycles of a directed graph.
       J.L. Szwarcfiter and P.E. Lauer, BIT NUMERICAL MATHEMATICS,
       v. 16, no. 2, 192-204, 1976.
    .. [5] Optimal Listing of Cycles and st-Paths in Undirected Graphs
        R. Ferreira and R. Grossi and A. Marino and N. Pisanti and R. Rizzi and
        G. Sacomoto https://arxiv.org/abs/1205.2766

    See Also
    --------
    cycle_basis
    chordless_cycles
    """
    if length_bound is not None:
        if length_bound == 0:
            return
        elif length_bound < 0:
            raise ValueError('length bound must be non-negative')
    directed = G.is_directed()
    yield from ([v] for v, Gv in G.adj.items() if v in Gv)
    if length_bound is not None and length_bound == 1:
        return
    if G.is_multigraph() and (not directed):
        visited = set()
        for u, Gu in G.adj.items():
            multiplicity = ((v, len(Guv)) for v, Guv in Gu.items() if v in visited)
            yield from ([u, v] for v, m in multiplicity if m > 1)
            visited.add(u)
    if directed:
        G = nx.DiGraph(((u, v) for u, Gu in G.adj.items() for v in Gu if v != u))
    else:
        G = nx.Graph(((u, v) for u, Gu in G.adj.items() for v in Gu if v != u))
    if length_bound is not None and length_bound == 2:
        if directed:
            visited = set()
            for u, Gu in G.adj.items():
                yield from ([v, u] for v in visited.intersection(Gu) if G.has_edge(v, u))
                visited.add(u)
        return
    if directed:
        yield from _directed_cycle_search(G, length_bound)
    else:
        yield from _undirected_cycle_search(G, length_bound)