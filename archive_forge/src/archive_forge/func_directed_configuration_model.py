import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
@py_random_state(3)
@nx._dispatch(graphs=None)
def directed_configuration_model(in_degree_sequence, out_degree_sequence, create_using=None, seed=None):
    """Returns a directed_random graph with the given degree sequences.

    The configuration model generates a random directed pseudograph
    (graph with parallel edges and self loops) by randomly assigning
    edges to match the given degree sequences.

    Parameters
    ----------
    in_degree_sequence :  list of nonnegative integers
       Each list entry corresponds to the in-degree of a node.
    out_degree_sequence :  list of nonnegative integers
       Each list entry corresponds to the out-degree of a node.
    create_using : NetworkX graph constructor, optional (default MultiDiGraph)
        Graph type to create. If graph instance, then cleared before populated.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : MultiDiGraph
        A graph with the specified degree sequences.
        Nodes are labeled starting at 0 with an index
        corresponding to the position in deg_sequence.

    Raises
    ------
    NetworkXError
        If the degree sequences do not have the same sum.

    See Also
    --------
    configuration_model

    Notes
    -----
    Algorithm as described by Newman [1]_.

    A non-graphical degree sequence (not realizable by some simple
    graph) is allowed since this function returns graphs with self
    loops and parallel edges.  An exception is raised if the degree
    sequences does not have the same sum.

    This configuration model construction process can lead to
    duplicate edges and loops.  You can remove the self-loops and
    parallel edges (see below) which will likely result in a graph
    that doesn't have the exact degree sequence specified.  This
    "finite-size effect" decreases as the size of the graph increases.

    References
    ----------
    .. [1] Newman, M. E. J. and Strogatz, S. H. and Watts, D. J.
       Random graphs with arbitrary degree distributions and their applications
       Phys. Rev. E, 64, 026118 (2001)

    Examples
    --------
    One can modify the in- and out-degree sequences from an existing
    directed graph in order to create a new directed graph. For example,
    here we modify the directed path graph:

    >>> D = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    >>> din = list(d for n, d in D.in_degree())
    >>> dout = list(d for n, d in D.out_degree())
    >>> din.append(1)
    >>> dout[0] = 2
    >>> # We now expect an edge from node 0 to a new node, node 3.
    ... D = nx.directed_configuration_model(din, dout)

    The returned graph is a directed multigraph, which may have parallel
    edges. To remove any parallel edges from the returned graph:

    >>> D = nx.DiGraph(D)

    Similarly, to remove self-loops:

    >>> D.remove_edges_from(nx.selfloop_edges(D))

    """
    if sum(in_degree_sequence) != sum(out_degree_sequence):
        msg = 'Invalid degree sequences: sequences must have equal sums'
        raise nx.NetworkXError(msg)
    if create_using is None:
        create_using = nx.MultiDiGraph
    G = _configuration_model(out_degree_sequence, create_using, directed=True, in_deg_sequence=in_degree_sequence, seed=seed)
    name = 'directed configuration_model {} nodes {} edges'
    return G