import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
@py_random_state(1)
@nx._dispatch(graphs=None)
def expected_degree_graph(w, seed=None, selfloops=True):
    """Returns a random graph with given expected degrees.

    Given a sequence of expected degrees $W=(w_0,w_1,\\ldots,w_{n-1})$
    of length $n$ this algorithm assigns an edge between node $u$ and
    node $v$ with probability

    .. math::

       p_{uv} = \\frac{w_u w_v}{\\sum_k w_k} .

    Parameters
    ----------
    w : list
        The list of expected degrees.
    selfloops: bool (default=True)
        Set to False to remove the possibility of self-loop edges.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    Graph

    Examples
    --------
    >>> z = [10 for i in range(100)]
    >>> G = nx.expected_degree_graph(z)

    Notes
    -----
    The nodes have integer labels corresponding to index of expected degrees
    input sequence.

    The complexity of this algorithm is $\\mathcal{O}(n+m)$ where $n$ is the
    number of nodes and $m$ is the expected number of edges.

    The model in [1]_ includes the possibility of self-loop edges.
    Set selfloops=False to produce a graph without self loops.

    For finite graphs this model doesn't produce exactly the given
    expected degree sequence.  Instead the expected degrees are as
    follows.

    For the case without self loops (selfloops=False),

    .. math::

       E[deg(u)] = \\sum_{v \\ne u} p_{uv}
                = w_u \\left( 1 - \\frac{w_u}{\\sum_k w_k} \\right) .


    NetworkX uses the standard convention that a self-loop edge counts 2
    in the degree of a node, so with self loops (selfloops=True),

    .. math::

       E[deg(u)] =  \\sum_{v \\ne u} p_{uv}  + 2 p_{uu}
                = w_u \\left( 1 + \\frac{w_u}{\\sum_k w_k} \\right) .

    References
    ----------
    .. [1] Fan Chung and L. Lu, Connected components in random graphs with
       given expected degree sequences, Ann. Combinatorics, 6,
       pp. 125-145, 2002.
    .. [2] Joel Miller and Aric Hagberg,
       Efficient generation of networks with given expected degrees,
       in Algorithms and Models for the Web-Graph (WAW 2011),
       Alan Frieze, Paul Horn, and Paweł Prałat (Eds), LNCS 6732,
       pp. 115-126, 2011.
    """
    n = len(w)
    G = nx.empty_graph(n)
    if n == 0 or max(w) == 0:
        return G
    rho = 1 / sum(w)
    order = sorted(enumerate(w), key=itemgetter(1), reverse=True)
    mapping = {c: u for c, (u, v) in enumerate(order)}
    seq = [v for u, v in order]
    last = n
    if not selfloops:
        last -= 1
    for u in range(last):
        v = u
        if not selfloops:
            v += 1
        factor = seq[u] * rho
        p = min(seq[v] * factor, 1)
        while v < n and p > 0:
            if p != 1:
                r = seed.random()
                v += math.floor(math.log(r, 1 - p))
            if v < n:
                q = min(seq[v] * factor, 1)
                if seed.random() < q / p:
                    G.add_edge(mapping[u], mapping[v])
                v += 1
                p = q
    return G