import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
@py_random_state(4)
@nx._dispatch(graphs=None)
def dual_barabasi_albert_graph(n, m1, m2, p, seed=None, initial_graph=None):
    """Returns a random graph using dual Barabási–Albert preferential attachment

    A graph of $n$ nodes is grown by attaching new nodes each with either $m_1$
    edges (with probability $p$) or $m_2$ edges (with probability $1-p$) that
    are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m1 : int
        Number of edges to link each new node to existing nodes with probability $p$
    m2 : int
        Number of edges to link each new node to existing nodes with probability $1-p$
    p : float
        The probability of attaching $m_1$ edges (as opposed to $m_2$ edges)
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    initial_graph : Graph or None (default)
        Initial network for Barabási–Albert algorithm.
        A copy of `initial_graph` is used.
        It should be connected for most use cases.
        If None, starts from an star graph on max(m1, m2) + 1 nodes.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m1` and `m2` do not satisfy ``1 <= m1,m2 < n``, or
        `p` does not satisfy ``0 <= p <= 1``, or
        the initial graph number of nodes m0 does not satisfy m1, m2 <= m0 <= n.

    References
    ----------
    .. [1] N. Moshiri "The dual-Barabasi-Albert model", arXiv:1810.10538.
    """
    if m1 < 1 or m1 >= n:
        raise nx.NetworkXError(f'Dual Barabási–Albert must have m1 >= 1 and m1 < n, m1 = {m1}, n = {n}')
    if m2 < 1 or m2 >= n:
        raise nx.NetworkXError(f'Dual Barabási–Albert must have m2 >= 1 and m2 < n, m2 = {m2}, n = {n}')
    if p < 0 or p > 1:
        raise nx.NetworkXError(f'Dual Barabási–Albert network must have 0 <= p <= 1, p = {p}')
    if p == 1:
        return barabasi_albert_graph(n, m1, seed)
    elif p == 0:
        return barabasi_albert_graph(n, m2, seed)
    if initial_graph is None:
        G = star_graph(max(m1, m2))
    else:
        if len(initial_graph) < max(m1, m2) or len(initial_graph) > n:
            raise nx.NetworkXError(f'Barabási–Albert initial graph must have between max(m1, m2) = {max(m1, m2)} and n = {n} nodes')
        G = initial_graph.copy()
    targets = list(G)
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    source = len(G)
    while source < n:
        if seed.random() < p:
            m = m1
        else:
            m = m2
        targets = _random_subset(repeated_nodes, m, seed)
        G.add_edges_from(zip([source] * m, targets))
        repeated_nodes.extend(targets)
        repeated_nodes.extend([source] * m)
        source += 1
    return G