import warnings
from collections.abc import Collection, Generator, Iterator
import networkx as nx
def to_dict_of_dicts(G, nodelist=None, edge_data=None):
    """Returns adjacency representation of graph as a dictionary of dictionaries.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list
       Use only nodes specified in nodelist

    edge_data : scalar, optional
       If provided, the value of the dictionary will be set to `edge_data` for
       all edges. Usual values could be `1` or `True`. If `edge_data` is
       `None` (the default), the edgedata in `G` is used, resulting in a
       dict-of-dict-of-dicts. If `G` is a MultiGraph, the result will be a
       dict-of-dict-of-dict-of-dicts. See Notes for an approach to customize
       handling edge data. `edge_data` should *not* be a container.

    Returns
    -------
    dod : dict
       A nested dictionary representation of `G`. Note that the level of
       nesting depends on the type of `G` and the value of `edge_data`
       (see Examples).

    See Also
    --------
    from_dict_of_dicts, to_dict_of_lists

    Notes
    -----
    For a more custom approach to handling edge data, try::

        dod = {
            n: {
                nbr: custom(n, nbr, dd) for nbr, dd in nbrdict.items()
            }
            for n, nbrdict in G.adj.items()
        }

    where `custom` returns the desired edge data for each edge between `n` and
    `nbr`, given existing edge data `dd`.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> nx.to_dict_of_dicts(G)
    {0: {1: {}}, 1: {0: {}, 2: {}}, 2: {1: {}}}

    Edge data is preserved by default (``edge_data=None``), resulting
    in dict-of-dict-of-dicts where the innermost dictionary contains the
    edge data:

    >>> G = nx.Graph()
    >>> G.add_edges_from(
    ...     [
    ...         (0, 1, {'weight': 1.0}),
    ...         (1, 2, {'weight': 2.0}),
    ...         (2, 0, {'weight': 1.0}),
    ...     ]
    ... )
    >>> d = nx.to_dict_of_dicts(G)
    >>> d  # doctest: +SKIP
    {0: {1: {'weight': 1.0}, 2: {'weight': 1.0}},
     1: {0: {'weight': 1.0}, 2: {'weight': 2.0}},
     2: {1: {'weight': 2.0}, 0: {'weight': 1.0}}}
    >>> d[1][2]['weight']
    2.0

    If `edge_data` is not `None`, edge data in the original graph (if any) is
    replaced:

    >>> d = nx.to_dict_of_dicts(G, edge_data=1)
    >>> d
    {0: {1: 1, 2: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 0: 1}}
    >>> d[1][2]
    1

    This also applies to MultiGraphs: edge data is preserved by default:

    >>> G = nx.MultiGraph()
    >>> G.add_edge(0, 1, key='a', weight=1.0)
    'a'
    >>> G.add_edge(0, 1, key='b', weight=5.0)
    'b'
    >>> d = nx.to_dict_of_dicts(G)
    >>> d  # doctest: +SKIP
    {0: {1: {'a': {'weight': 1.0}, 'b': {'weight': 5.0}}},
     1: {0: {'a': {'weight': 1.0}, 'b': {'weight': 5.0}}}}
    >>> d[0][1]['b']['weight']
    5.0

    But multi edge data is lost if `edge_data` is not `None`:

    >>> d = nx.to_dict_of_dicts(G, edge_data=10)
    >>> d
    {0: {1: 10}, 1: {0: 10}}
    """
    dod = {}
    if nodelist is None:
        if edge_data is None:
            for u, nbrdict in G.adjacency():
                dod[u] = nbrdict.copy()
        else:
            for u, nbrdict in G.adjacency():
                dod[u] = dod.fromkeys(nbrdict, edge_data)
    elif edge_data is None:
        for u in nodelist:
            dod[u] = {}
            for v, data in ((v, data) for v, data in G[u].items() if v in nodelist):
                dod[u][v] = data
    else:
        for u in nodelist:
            dod[u] = {}
            for v in (v for v in G[u] if v in nodelist):
                dod[u][v] = edge_data
    return dod