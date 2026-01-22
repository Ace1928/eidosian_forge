import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
@nx._dispatch(graphs=None)
def directed_havel_hakimi_graph(in_deg_sequence, out_deg_sequence, create_using=None):
    """Returns a directed graph with the given degree sequences.

    Parameters
    ----------
    in_deg_sequence :  list of integers
        Each list entry corresponds to the in-degree of a node.
    out_deg_sequence : list of integers
        Each list entry corresponds to the out-degree of a node.
    create_using : NetworkX graph constructor, optional (default DiGraph)
        Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : DiGraph
        A graph with the specified degree sequences.
        Nodes are labeled starting at 0 with an index
        corresponding to the position in deg_sequence

    Raises
    ------
    NetworkXError
        If the degree sequences are not digraphical.

    See Also
    --------
    configuration_model

    Notes
    -----
    Algorithm as described by Kleitman and Wang [1]_.

    References
    ----------
    .. [1] D.J. Kleitman and D.L. Wang
       Algorithms for Constructing Graphs and Digraphs with Given Valences
       and Factors Discrete Mathematics, 6(1), pp. 79-88 (1973)
    """
    in_deg_sequence = nx.utils.make_list_of_ints(in_deg_sequence)
    out_deg_sequence = nx.utils.make_list_of_ints(out_deg_sequence)
    sumin, sumout = (0, 0)
    nin, nout = (len(in_deg_sequence), len(out_deg_sequence))
    maxn = max(nin, nout)
    G = nx.empty_graph(maxn, create_using, default=nx.DiGraph)
    if maxn == 0:
        return G
    maxin = 0
    stubheap, zeroheap = ([], [])
    for n in range(maxn):
        in_deg, out_deg = (0, 0)
        if n < nout:
            out_deg = out_deg_sequence[n]
        if n < nin:
            in_deg = in_deg_sequence[n]
        if in_deg < 0 or out_deg < 0:
            raise nx.NetworkXError('Invalid degree sequences. Sequence values must be positive.')
        sumin, sumout, maxin = (sumin + in_deg, sumout + out_deg, max(maxin, in_deg))
        if in_deg > 0:
            stubheap.append((-1 * out_deg, -1 * in_deg, n))
        elif out_deg > 0:
            zeroheap.append((-1 * out_deg, n))
    if sumin != sumout:
        raise nx.NetworkXError('Invalid degree sequences. Sequences must have equal sums.')
    heapq.heapify(stubheap)
    heapq.heapify(zeroheap)
    modstubs = [(0, 0, 0)] * (maxin + 1)
    while stubheap:
        freeout, freein, target = heapq.heappop(stubheap)
        freein *= -1
        if freein > len(stubheap) + len(zeroheap):
            raise nx.NetworkXError('Non-digraphical integer sequence')
        mslen = 0
        for i in range(freein):
            if zeroheap and (not stubheap or stubheap[0][0] > zeroheap[0][0]):
                stubout, stubsource = heapq.heappop(zeroheap)
                stubin = 0
            else:
                stubout, stubin, stubsource = heapq.heappop(stubheap)
            if stubout == 0:
                raise nx.NetworkXError('Non-digraphical integer sequence')
            G.add_edge(stubsource, target)
            if stubout + 1 < 0 or stubin < 0:
                modstubs[mslen] = (stubout + 1, stubin, stubsource)
                mslen += 1
        for i in range(mslen):
            stub = modstubs[i]
            if stub[1] < 0:
                heapq.heappush(stubheap, stub)
            else:
                heapq.heappush(zeroheap, (stub[0], stub[2]))
        if freeout < 0:
            heapq.heappush(zeroheap, (freeout, target))
    return G