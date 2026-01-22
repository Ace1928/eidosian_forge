from itertools import chain
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def edge_expansion(G, S, T=None, weight=None):
    """Returns the edge expansion between two node sets.

    The *edge expansion* is the quotient of the cut size and the smaller
    of the cardinalities of the two sets. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The edge expansion between the two sets `S` and `T`.

    See also
    --------
    boundary_expansion
    mixing_expansion
    node_expansion

    References
    ----------
    .. [1] Fan Chung.
           *Spectral Graph Theory*.
           (CBMS Regional Conference Series in Mathematics, No. 92),
           American Mathematical Society, 1997, ISBN 0-8218-0315-8
           <http://www.math.ucsd.edu/~fan/research/revised.html>

    """
    if T is None:
        T = set(G) - set(S)
    num_cut_edges = cut_size(G, S, T=T, weight=weight)
    return num_cut_edges / min(len(S), len(T))