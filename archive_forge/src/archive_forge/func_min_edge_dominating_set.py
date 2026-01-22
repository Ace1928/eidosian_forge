import networkx as nx
from ...utils import not_implemented_for
from ..matching import maximal_matching
@nx._dispatch
def min_edge_dominating_set(G):
    """Returns minimum cardinality edge dominating set.

    Parameters
    ----------
    G : NetworkX graph
      Undirected graph

    Returns
    -------
    min_edge_dominating_set : set
      Returns a set of dominating edges whose size is no more than 2 * OPT.

    Notes
    -----
    The algorithm computes an approximate solution to the edge dominating set
    problem. The result is no more than 2 * OPT in terms of size of the set.
    Runtime of the algorithm is $O(|E|)$.
    """
    if not G:
        raise ValueError('Expected non-empty NetworkX graph!')
    return maximal_matching(G)