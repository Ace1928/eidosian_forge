from itertools import islice
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for, open_file
@not_implemented_for('directed')
@not_implemented_for('multigraph')
def write_graph6_file(G, f, nodes=None, header=True):
    """Write a simple undirected graph to a file-like object in graph6 format.

    Parameters
    ----------
    G : Graph (undirected)

    f : file-like object
       The file to write.

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by ``G.nodes()`` is used.

    header: bool
       If True add '>>graph6<<' string to head of data

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    ValueError
        If the graph has at least ``2 ** 36`` nodes; the graph6 format
        is only defined for graphs of order less than ``2 ** 36``.

    Examples
    --------
    You can write a graph6 file by giving an open file-like object::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     nx.write_graph6(nx.path_graph(2), f)
        ...     _ = f.seek(0)
        ...     print(f.read())
        b'>>graph6<<A_\\n'

    See Also
    --------
    from_graph6_bytes, read_graph6

    Notes
    -----
    The function writes a newline character after writing the encoding
    of the graph.

    The format does not support edge or node labels, parallel edges or
    self loops.  If self loops are present they are silently ignored.

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    if nodes is not None:
        G = G.subgraph(nodes)
    H = nx.convert_node_labels_to_integers(G)
    nodes = sorted(H.nodes())
    for b in _generate_graph6_bytes(H, nodes, header):
        f.write(b)