import networkx as nx
from networkx.utils import not_implemented_for, open_file
@open_file(0, mode='rb')
@nx._dispatch(name='bipartite_read_edgelist', graphs=None)
def read_edgelist(path, comments='#', delimiter=None, create_using=None, nodetype=None, data=True, edgetype=None, encoding='utf-8'):
    """Read a bipartite graph from a list of edges.

    Parameters
    ----------
    path : file or string
       File or filename to read. If a file is provided, it must be
       opened in 'rb' mode.
       Filenames ending in .gz or .bz2 will be uncompressed.
    comments : string, optional
       The character used to indicate the start of a comment.
    delimiter : string, optional
       The string used to separate values.  The default is whitespace.
    create_using : Graph container, optional,
       Use specified container to build graph.  The default is networkx.Graph,
       an undirected graph.
    nodetype : int, float, str, Python type, optional
       Convert node data from strings to specified type
    data : bool or list of (label,type) tuples
       Tuples specifying dictionary key names and types for edge data
    edgetype : int, float, str, Python type, optional OBSOLETE
       Convert edge data from strings to specified type and use as 'weight'
    encoding: string, optional
       Specify which encoding to use when reading file.

    Returns
    -------
    G : graph
       A networkx Graph or other type specified with create_using

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> G.add_nodes_from([0, 2], bipartite=0)
    >>> G.add_nodes_from([1, 3], bipartite=1)
    >>> bipartite.write_edgelist(G, "test.edgelist")
    >>> G = bipartite.read_edgelist("test.edgelist")

    >>> fh = open("test.edgelist", "rb")
    >>> G = bipartite.read_edgelist(fh)
    >>> fh.close()

    >>> G = bipartite.read_edgelist("test.edgelist", nodetype=int)

    Edgelist with data in a list:

    >>> textline = "1 2 3"
    >>> fh = open("test.edgelist", "w")
    >>> d = fh.write(textline)
    >>> fh.close()
    >>> G = bipartite.read_edgelist(
    ...     "test.edgelist", nodetype=int, data=(("weight", float),)
    ... )
    >>> list(G)
    [1, 2]
    >>> list(G.edges(data=True))
    [(1, 2, {'weight': 3.0})]

    See parse_edgelist() for more examples of formatting.

    See Also
    --------
    parse_edgelist

    Notes
    -----
    Since nodes must be hashable, the function nodetype must return hashable
    types (e.g. int, float, str, frozenset - or tuples of those, etc.)
    """
    lines = (line.decode(encoding) for line in path)
    return parse_edgelist(lines, comments=comments, delimiter=delimiter, create_using=create_using, nodetype=nodetype, data=data)