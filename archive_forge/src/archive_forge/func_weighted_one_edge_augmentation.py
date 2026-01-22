import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@nx._dispatch
def weighted_one_edge_augmentation(G, avail, weight=None, partial=False):
    """Finds the minimum weight set of edges to connect G if one exists.

    This is a variant of the weighted MST problem.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    partial : boolean
        If partial is True and no feasible k-edge-augmentation exists, then the
        augmenting edges minimize the number of connected components.

    Yields
    ------
    edge : tuple
        Edges in the subset of avail chosen to connect G.

    See Also
    --------
    :func:`one_edge_augmentation`
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3), (4, 5)])
    >>> G.add_nodes_from([6, 7, 8])
    >>> # any edge not in avail has an implicit weight of infinity
    >>> avail = [(1, 3), (1, 5), (4, 7), (4, 8), (6, 1), (8, 1), (8, 2)]
    >>> sorted(weighted_one_edge_augmentation(G, avail))
    [(1, 5), (4, 7), (6, 1), (8, 1)]
    >>> # find another solution by giving large weights to edges in the
    >>> # previous solution (note some of the old edges must be used)
    >>> avail = [(1, 3), (1, 5, 99), (4, 7, 9), (6, 1, 99), (8, 1, 99), (8, 2)]
    >>> sorted(weighted_one_edge_augmentation(G, avail))
    [(1, 5), (4, 7), (6, 1), (8, 2)]
    """
    avail_uv, avail_w = _unpack_available_edges(avail, weight=weight, G=G)
    C = collapse(G, nx.connected_components(G))
    mapping = C.graph['mapping']
    candidate_mapping = _lightest_meta_edges(mapping, avail_uv, avail_w)
    C.add_edges_from(((mu, mv, {'weight': w, 'generator': uv}) for (mu, mv), uv, w in candidate_mapping))
    meta_mst = nx.minimum_spanning_tree(C)
    if not partial and (not nx.is_connected(meta_mst)):
        raise nx.NetworkXUnfeasible('Not possible to connect G with available edges')
    for mu, mv, d in meta_mst.edges(data=True):
        if 'generator' in d:
            edge = d['generator']
            yield edge