import itertools
from collections import defaultdict, deque
import networkx as nx
from networkx.utils import arbitrary_element, py_random_state
@nx._dispatch
def strategy_connected_sequential(G, colors, traversal='bfs'):
    """Returns an iterable over nodes in ``G`` in the order given by a
    breadth-first or depth-first traversal.

    ``traversal`` must be one of the strings ``'dfs'`` or ``'bfs'``,
    representing depth-first traversal or breadth-first traversal,
    respectively.

    The generated sequence has the property that for each node except
    the first, at least one neighbor appeared earlier in the sequence.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    """
    if traversal == 'bfs':
        traverse = nx.bfs_edges
    elif traversal == 'dfs':
        traverse = nx.dfs_edges
    else:
        raise nx.NetworkXError("Please specify one of the strings 'bfs' or 'dfs' for connected sequential ordering")
    for component in nx.connected_components(G):
        source = arbitrary_element(component)
        yield source
        for _, end in traverse(G.subgraph(component), source):
            yield end