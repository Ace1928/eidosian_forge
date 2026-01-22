from itertools import islice
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for, open_file
@nx._dispatch(graphs=None)
def from_graph6_bytes(bytes_in):
    """Read a simple undirected graph in graph6 format from bytes.

    Parameters
    ----------
    bytes_in : bytes
       Data in graph6 format, without a trailing newline.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If bytes_in is unable to be parsed in graph6 format

    ValueError
        If any character ``c`` in bytes_in does not satisfy
        ``63 <= ord(c) < 127``.

    Examples
    --------
    >>> G = nx.from_graph6_bytes(b"A_")
    >>> sorted(G.edges())
    [(0, 1)]

    See Also
    --------
    read_graph6, write_graph6

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """

    def bits():
        """Returns sequence of individual bits from 6-bit-per-value
        list of data values."""
        for d in data:
            for i in [5, 4, 3, 2, 1, 0]:
                yield (d >> i & 1)
    if bytes_in.startswith(b'>>graph6<<'):
        bytes_in = bytes_in[10:]
    data = [c - 63 for c in bytes_in]
    if any((c > 63 for c in data)):
        raise ValueError('each input character must be in range(63, 127)')
    n, data = data_to_n(data)
    nd = (n * (n - 1) // 2 + 5) // 6
    if len(data) != nd:
        raise NetworkXError(f'Expected {n * (n - 1) // 2} bits but got {len(data) * 6} in graph6')
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), b in zip(((i, j) for j in range(1, n) for i in range(j)), bits()):
        if b:
            G.add_edge(i, j)
    return G