import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
@nx._dispatch(graphs=None)
def parse_leda(lines):
    """Read graph in LEDA format from string or iterable.

    Parameters
    ----------
    lines : string or iterable
       Data in LEDA format.

    Returns
    -------
    G : NetworkX graph

    Examples
    --------
    G=nx.parse_leda(string)

    References
    ----------
    .. [1] http://www.algorithmic-solutions.info/leda_guide/graphs/leda_native_graph_fileformat.html
    """
    if isinstance(lines, str):
        lines = iter(lines.split('\n'))
    lines = iter([line.rstrip('\n') for line in lines if not (line.startswith(('#', '\n')) or line == '')])
    for i in range(3):
        next(lines)
    du = int(next(lines))
    if du == -1:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    n = int(next(lines))
    node = {}
    for i in range(1, n + 1):
        symbol = next(lines).rstrip().strip('|{}|  ')
        if symbol == '':
            symbol = str(i)
        node[i] = symbol
    G.add_nodes_from([s for i, s in node.items()])
    m = int(next(lines))
    for i in range(m):
        try:
            s, t, reversal, label = next(lines).split()
        except BaseException as err:
            raise NetworkXError(f'Too few fields in LEDA.GRAPH edge {i + 1}') from err
        G.add_edge(node[int(s)], node[int(t)], label=label[2:-2])
    return G