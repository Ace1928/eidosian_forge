import networkx as nx
from networkx.utils import not_implemented_for, open_file
@nx._dispatch(name='bipartite_parse_edgelist', graphs=None)
def parse_edgelist(lines, comments='#', delimiter=None, create_using=None, nodetype=None, data=True):
    """Parse lines of an edge list representation of a bipartite graph.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in edgelist format
    comments : string, optional
       Marker for comment lines
    delimiter : string, optional
       Separator for node labels
    create_using: NetworkX graph container, optional
       Use given NetworkX graph for holding nodes or edges.
    nodetype : Python type, optional
       Convert nodes to this type.
    data : bool or list of (label,type) tuples
       If False generate no edge data or if True use a dictionary
       representation of edge data or a list tuples specifying dictionary
       key names and types for edge data.

    Returns
    -------
    G: NetworkX Graph
        The bipartite graph corresponding to lines

    Examples
    --------
    Edgelist with no data:

    >>> from networkx.algorithms import bipartite
    >>> lines = ["1 2", "2 3", "3 4"]
    >>> G = bipartite.parse_edgelist(lines, nodetype=int)
    >>> sorted(G.nodes())
    [1, 2, 3, 4]
    >>> sorted(G.nodes(data=True))
    [(1, {'bipartite': 0}), (2, {'bipartite': 0}), (3, {'bipartite': 0}), (4, {'bipartite': 1})]
    >>> sorted(G.edges())
    [(1, 2), (2, 3), (3, 4)]

    Edgelist with data in Python dictionary representation:

    >>> lines = ["1 2 {'weight':3}", "2 3 {'weight':27}", "3 4 {'weight':3.0}"]
    >>> G = bipartite.parse_edgelist(lines, nodetype=int)
    >>> sorted(G.nodes())
    [1, 2, 3, 4]
    >>> sorted(G.edges(data=True))
    [(1, 2, {'weight': 3}), (2, 3, {'weight': 27}), (3, 4, {'weight': 3.0})]

    Edgelist with data in a list:

    >>> lines = ["1 2 3", "2 3 27", "3 4 3.0"]
    >>> G = bipartite.parse_edgelist(lines, nodetype=int, data=(("weight", float),))
    >>> sorted(G.nodes())
    [1, 2, 3, 4]
    >>> sorted(G.edges(data=True))
    [(1, 2, {'weight': 3.0}), (2, 3, {'weight': 27.0}), (3, 4, {'weight': 3.0})]

    See Also
    --------
    """
    from ast import literal_eval
    G = nx.empty_graph(0, create_using)
    for line in lines:
        p = line.find(comments)
        if p >= 0:
            line = line[:p]
        if not len(line):
            continue
        s = line.strip().split(delimiter)
        if len(s) < 2:
            continue
        u = s.pop(0)
        v = s.pop(0)
        d = s
        if nodetype is not None:
            try:
                u = nodetype(u)
                v = nodetype(v)
            except BaseException as err:
                raise TypeError(f'Failed to convert nodes {u},{v} to type {nodetype}.') from err
        if len(d) == 0 or data is False:
            edgedata = {}
        elif data is True:
            try:
                edgedata = dict(literal_eval(' '.join(d)))
            except BaseException as err:
                raise TypeError(f'Failed to convert edge data ({d})to dictionary.') from err
        else:
            if len(d) != len(data):
                raise IndexError(f'Edge data {d} and data_keys {data} are not the same length')
            edgedata = {}
            for (edge_key, edge_type), edge_value in zip(data, d):
                try:
                    edge_value = edge_type(edge_value)
                except BaseException as err:
                    raise TypeError(f'Failed to convert {edge_key} data {edge_value} to type {edge_type}.') from err
                edgedata.update({edge_key: edge_value})
        G.add_node(u, bipartite=0)
        G.add_node(v, bipartite=1)
        G.add_edge(u, v, **edgedata)
    return G