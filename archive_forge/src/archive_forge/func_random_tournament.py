from itertools import combinations
import networkx as nx
from networkx.algorithms.simple_paths import is_simple_path as is_path
from networkx.utils import arbitrary_element, not_implemented_for, py_random_state
@py_random_state(1)
@nx._dispatch(graphs=None)
def random_tournament(n, seed=None):
    """Returns a random tournament graph on `n` nodes.

    Parameters
    ----------
    n : int
        The number of nodes in the returned graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : DiGraph
        A tournament on `n` nodes, with exactly one directed edge joining
        each pair of distinct nodes.

    Notes
    -----
    This algorithm adds, for each pair of distinct nodes, an edge with
    uniformly random orientation. In other words, `\\binom{n}{2}` flips
    of an unbiased coin decide the orientations of the edges in the
    graph.

    """
    coins = (seed.random() for i in range(n * (n - 1) // 2))
    pairs = combinations(range(n), 2)
    edges = ((u, v) if r < 0.5 else (v, u) for (u, v), r in zip(pairs, coins))
    return nx.DiGraph(edges)