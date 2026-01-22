import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
@py_random_state('seed')
@nx._dispatch(graphs=None)
def random_labeled_rooted_forest(n, *, seed=None):
    """Returns a labeled rooted forest with `n` nodes.

    The returned forest is chosen uniformly at random using a
    generalization of Prüfer sequences [1]_ in the form described in [2]_.

    Parameters
    ----------
    n : int
        The number of nodes.
    seed : random_state
       See :ref:`Randomness<randomness>`.

    Returns
    -------
    :class:`networkx.Graph`
        A `networkx.Graph` with integer nodes 0 <= node <= `n` - 1.
        The "roots" graph attribute is a set of integers containing the roots.

    References
    ----------
    .. [1] Knuth, Donald E. "Another Enumeration of Trees."
        Canadian Journal of Mathematics, 20 (1968): 1077-1086.
        https://doi.org/10.4153/CJM-1968-104-8
    .. [2] Rubey, Martin. "Counting Spanning Trees". Diplomarbeit
        zur Erlangung des akademischen Grades Magister der
        Naturwissenschaften an der Formal- und Naturwissenschaftlichen
        Fakultät der Universität Wien. Wien, May 2000.
    """

    def _select_k(n, seed):
        r = seed.randint(0, (n + 1) ** (n - 1) - 1)
        cum_sum = 0
        for k in range(1, n):
            cum_sum += factorial(n - 1) * n ** (n - k) // (factorial(k - 1) * factorial(n - k))
            if r < cum_sum:
                return k
        return n
    F = nx.empty_graph(n)
    if n == 0:
        F.graph['roots'] = {}
        return F
    k = _select_k(n, seed)
    if k == n:
        F.graph['roots'] = set(range(n))
        return F
    roots = seed.sample(range(n), k)
    p = set(range(n)).difference(roots)
    N = [seed.randint(0, n - 1) for i in range(n - k - 1)]
    degree = Counter([x for x in N if x in p])
    iterator = iter((x for x in p if degree[x] == 0))
    u = last = next(iterator)
    for v in N:
        F.add_edge(u, v)
        degree[v] -= 1
        if v < last and degree[v] == 0:
            u = v
        else:
            last = u = next(iterator)
    F.add_edge(u, roots[0])
    F.graph['roots'] = set(roots)
    return F