import heapq
import networkx as nx
@nx._dispatch(graphs=None)
def is_pseudographical(sequence):
    """Returns True if some pseudograph can realize the sequence.

    Every nonnegative integer sequence with an even sum is pseudographical
    (see [1]_).

    Parameters
    ----------
    sequence : list or iterable container
        A sequence of integer node degrees

    Returns
    -------
    valid : bool
      True if the sequence is a pseudographic degree sequence and False if not.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])
    >>> sequence = (d for _, d in G.degree())
    >>> nx.is_pseudographical(sequence)
    True

    To test a non-pseudographical sequence:
    >>> sequence_list = [d for _, d in G.degree()]
    >>> sequence_list[-1] += 1
    >>> nx.is_pseudographical(sequence_list)
    False

    Notes
    -----
    The worst-case run time is $O(n)$ where n is the length of the sequence.

    References
    ----------
    .. [1] F. Boesch and F. Harary. "Line removal algorithms for graphs
       and their degree lists", IEEE Trans. Circuits and Systems, CAS-23(12),
       pp. 778-782 (1976).
    """
    try:
        deg_sequence = nx.utils.make_list_of_ints(sequence)
    except nx.NetworkXError:
        return False
    return sum(deg_sequence) % 2 == 0 and min(deg_sequence) >= 0