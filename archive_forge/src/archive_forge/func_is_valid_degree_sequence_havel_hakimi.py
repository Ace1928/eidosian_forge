import heapq
import networkx as nx
@nx._dispatch(graphs=None)
def is_valid_degree_sequence_havel_hakimi(deg_sequence):
    """Returns True if deg_sequence can be realized by a simple graph.

    The validation proceeds using the Havel-Hakimi theorem
    [havel1955]_, [hakimi1962]_, [CL1996]_.
    Worst-case run time is $O(s)$ where $s$ is the sum of the sequence.

    Parameters
    ----------
    deg_sequence : list
        A list of integers where each element specifies the degree of a node
        in a graph.

    Returns
    -------
    valid : bool
        True if deg_sequence is graphical and False if not.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])
    >>> sequence = (d for _, d in G.degree())
    >>> nx.is_valid_degree_sequence_havel_hakimi(sequence)
    True

    To test a non-valid sequence:
    >>> sequence_list = [d for _, d in G.degree()]
    >>> sequence_list[-1] += 1
    >>> nx.is_valid_degree_sequence_havel_hakimi(sequence_list)
    False

    Notes
    -----
    The ZZ condition says that for the sequence d if

    .. math::
        |d| >= \\frac{(\\max(d) + \\min(d) + 1)^2}{4*\\min(d)}

    then d is graphical.  This was shown in Theorem 6 in [1]_.

    References
    ----------
    .. [1] I.E. Zverovich and V.E. Zverovich. "Contributions to the theory
       of graphic sequences", Discrete Mathematics, 105, pp. 292-303 (1992).
    .. [havel1955] Havel, V. "A Remark on the Existence of Finite Graphs"
       Casopis Pest. Mat. 80, 477-480, 1955.
    .. [hakimi1962] Hakimi, S. "On the Realizability of a Set of Integers as
       Degrees of the Vertices of a Graph." SIAM J. Appl. Math. 10, 496-506, 1962.
    .. [CL1996] G. Chartrand and L. Lesniak, "Graphs and Digraphs",
       Chapman and Hall/CRC, 1996.
    """
    try:
        dmax, dmin, dsum, n, num_degs = _basic_graphical_tests(deg_sequence)
    except nx.NetworkXUnfeasible:
        return False
    if n == 0 or 4 * dmin * n >= (dmax + dmin + 1) * (dmax + dmin + 1):
        return True
    modstubs = [0] * (dmax + 1)
    while n > 0:
        while num_degs[dmax] == 0:
            dmax -= 1
        if dmax > n - 1:
            return False
        num_degs[dmax], n = (num_degs[dmax] - 1, n - 1)
        mslen = 0
        k = dmax
        for i in range(dmax):
            while num_degs[k] == 0:
                k -= 1
            num_degs[k], n = (num_degs[k] - 1, n - 1)
            if k > 1:
                modstubs[mslen] = k - 1
                mslen += 1
        for i in range(mslen):
            stub = modstubs[i]
            num_degs[stub], n = (num_degs[stub] + 1, n + 1)
    return True