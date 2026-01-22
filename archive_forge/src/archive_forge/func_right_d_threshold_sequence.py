from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def right_d_threshold_sequence(n, m):
    """
    Create a skewed threshold graph with a given number
    of vertices (n) and a given number of edges (m).

    The routine returns an unlabeled creation sequence
    for the threshold graph.

    FIXME: describe algorithm

    """
    cs = ['d'] + ['i'] * (n - 1)
    if m < n:
        cs[m] = 'd'
        return cs
    if m > n * (n - 1) / 2:
        raise ValueError('Too many edges for this many nodes.')
    ind = n - 1
    sum = n - 1
    while sum < m:
        cs[ind] = 'd'
        ind -= 1
        sum += ind
    ind = m - (sum - ind)
    cs[ind] = 'd'
    return cs