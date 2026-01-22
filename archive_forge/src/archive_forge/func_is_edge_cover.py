from functools import partial
from itertools import chain
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
@not_implemented_for('directed')
@nx._dispatch
def is_edge_cover(G, cover):
    """Decides whether a set of edges is a valid edge cover of the graph.

    Given a set of edges, whether it is an edge covering can
    be decided if we just check whether all nodes of the graph
    has an edge from the set, incident on it.

    Parameters
    ----------
    G : NetworkX graph
        An undirected bipartite graph.

    cover : set
        Set of edges to be checked.

    Returns
    -------
    bool
        Whether the set of edges is a valid edge cover of the graph.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    >>> cover = {(2, 1), (3, 0)}
    >>> nx.is_edge_cover(G, cover)
    True

    Notes
    -----
    An edge cover of a graph is a set of edges such that every node of
    the graph is incident to at least one edge of the set.
    """
    return set(G) <= set(chain.from_iterable(cover))