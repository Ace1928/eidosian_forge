import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
@nx._dispatch
def is_aperiodic(G):
    """Returns True if `G` is aperiodic.

    A directed graph is aperiodic if there is no integer k > 1 that
    divides the length of every cycle in the graph.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed graph

    Returns
    -------
    bool
        True if the graph is aperiodic False otherwise

    Raises
    ------
    NetworkXError
        If `G` is not directed

    Examples
    --------
    A graph consisting of one cycle, the length of which is 2. Therefore ``k = 2``
    divides the length of every cycle in the graph and thus the graph
    is *not aperiodic*::

        >>> DG = nx.DiGraph([(1, 2), (2, 1)])
        >>> nx.is_aperiodic(DG)
        False

    A graph consisting of two cycles: one of length 2 and the other of length 3.
    The cycle lengths are coprime, so there is no single value of k where ``k > 1``
    that divides each cycle length and therefore the graph is *aperiodic*::

        >>> DG = nx.DiGraph([(1, 2), (2, 3), (3, 1), (1, 4), (4, 1)])
        >>> nx.is_aperiodic(DG)
        True

    A graph consisting of two cycles: one of length 2 and the other of length 4.
    The lengths of the cycles share a common factor ``k = 2``, and therefore
    the graph is *not aperiodic*::

        >>> DG = nx.DiGraph([(1, 2), (2, 1), (3, 4), (4, 5), (5, 6), (6, 3)])
        >>> nx.is_aperiodic(DG)
        False

    An acyclic graph, therefore the graph is *not aperiodic*::

        >>> DG = nx.DiGraph([(1, 2), (2, 3)])
        >>> nx.is_aperiodic(DG)
        False

    Notes
    -----
    This uses the method outlined in [1]_, which runs in $O(m)$ time
    given $m$ edges in `G`. Note that a graph is not aperiodic if it is
    acyclic as every integer trivial divides length 0 cycles.

    References
    ----------
    .. [1] Jarvis, J. P.; Shier, D. R. (1996),
       "Graph-theoretic analysis of finite Markov chains,"
       in Shier, D. R.; Wallenius, K. T., Applied Mathematical Modeling:
       A Multidisciplinary Approach, CRC Press.
    """
    if not G.is_directed():
        raise nx.NetworkXError('is_aperiodic not defined for undirected graphs')
    s = arbitrary_element(G)
    levels = {s: 0}
    this_level = [s]
    g = 0
    lev = 1
    while this_level:
        next_level = []
        for u in this_level:
            for v in G[u]:
                if v in levels:
                    g = gcd(g, levels[u] - levels[v] + 1)
                else:
                    next_level.append(v)
                    levels[v] = lev
        this_level = next_level
        lev += 1
    if len(levels) == len(G):
        return g == 1
    else:
        return g == 1 and nx.is_aperiodic(G.subgraph(set(G) - set(levels)))