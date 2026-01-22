import networkx as nx
from networkx.utils import open_file
@nx._dispatch(graphs=None)
def parse_adjlist(lines, comments='#', delimiter=None, create_using=None, nodetype=None):
    """Parse lines of a graph adjacency list representation.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in adjlist format

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    nodetype : Python type, optional
       Convert nodes to this type.

    comments : string, optional
       Marker for comment lines

    delimiter : string, optional
       Separator for node labels.  The default is whitespace.

    Returns
    -------
    G: NetworkX graph
        The graph corresponding to the lines in adjacency list format.

    Examples
    --------
    >>> lines = ["1 2 5", "2 3 4", "3 5", "4", "5"]
    >>> G = nx.parse_adjlist(lines, nodetype=int)
    >>> nodes = [1, 2, 3, 4, 5]
    >>> all(node in G for node in nodes)
    True
    >>> edges = [(1, 2), (1, 5), (2, 3), (2, 4), (3, 5)]
    >>> all((u, v) in G.edges() or (v, u) in G.edges() for (u, v) in edges)
    True

    See Also
    --------
    read_adjlist

    """
    G = nx.empty_graph(0, create_using)
    for line in lines:
        p = line.find(comments)
        if p >= 0:
            line = line[:p]
        if not len(line):
            continue
        vlist = line.strip().split(delimiter)
        u = vlist.pop(0)
        if nodetype is not None:
            try:
                u = nodetype(u)
            except BaseException as err:
                raise TypeError(f'Failed to convert node ({u}) to type {nodetype}') from err
        G.add_node(u)
        if nodetype is not None:
            try:
                vlist = list(map(nodetype, vlist))
            except BaseException as err:
                raise TypeError(f'Failed to convert nodes ({','.join(vlist)}) to type {nodetype}') from err
        G.add_edges_from([(u, v) for v in vlist])
    return G