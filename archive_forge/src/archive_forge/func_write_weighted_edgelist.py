import networkx as nx
from networkx.utils import open_file
def write_weighted_edgelist(G, path, comments='#', delimiter=' ', encoding='utf-8'):
    """Write graph G as a list of edges with numeric weights.

    Parameters
    ----------
    G : graph
       A NetworkX graph
    path : file or string
       File or filename to write. If a file is provided, it must be
       opened in 'wb' mode.
       Filenames ending in .gz or .bz2 will be compressed.
    comments : string, optional
       The character used to indicate the start of a comment
    delimiter : string, optional
       The string used to separate values.  The default is whitespace.
    encoding: string, optional
       Specify which encoding to use when writing file.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=7)
    >>> nx.write_weighted_edgelist(G, "test.weighted.edgelist")

    See Also
    --------
    read_edgelist
    write_edgelist
    read_weighted_edgelist
    """
    write_edgelist(G, path, comments=comments, delimiter=delimiter, data=('weight',), encoding=encoding)