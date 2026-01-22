from ``u`` to ``v`` that is not blocked. That means, there is an open
from collections import deque
from itertools import chain
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for
@not_implemented_for('undirected')
@nx._dispatchable
def is_d_separator(G, x, y, z):
    """Return whether node sets `x` and `y` are d-separated by `z`.

    Parameters
    ----------
    G : nx.DiGraph
        A NetworkX DAG.

    x : node or set of nodes
        First node or set of nodes in `G`.

    y : node or set of nodes
        Second node or set of nodes in `G`.

    z : node or set of nodes
        Potential separator (set of conditioning nodes in `G`). Can be empty set.

    Returns
    -------
    b : bool
        A boolean that is true if `x` is d-separated from `y` given `z` in `G`.

    Raises
    ------
    NetworkXError
        The *d-separation* test is commonly used on disjoint sets of
        nodes in acyclic directed graphs.  Accordingly, the algorithm
        raises a :exc:`NetworkXError` if the node sets are not
        disjoint or if the input graph is not a DAG.

    NodeNotFound
        If any of the input nodes are not found in the graph,
        a :exc:`NodeNotFound` exception is raised

    Notes
    -----
    A d-separating set in a DAG is a set of nodes that
    blocks all paths between the two sets. Nodes in `z`
    block a path if they are part of the path and are not a collider,
    or a descendant of a collider. Also colliders that are not in `z`
    block a path. A collider structure along a path
    is ``... -> c <- ...`` where ``c`` is the collider node.

    https://en.wikipedia.org/wiki/Bayesian_network#d-separation
    """
    try:
        x = {x} if x in G else x
        y = {y} if y in G else y
        z = {z} if z in G else z
        intersection = x & y or x & z or y & z
        if intersection:
            raise nx.NetworkXError(f'The sets are not disjoint, with intersection {intersection}')
        set_v = x | y | z
        if set_v - G.nodes:
            raise nx.NodeNotFound(f'The node(s) {set_v - G.nodes} are not found in G')
    except TypeError:
        raise nx.NodeNotFound('One of x, y, or z is not a node or a set of nodes in G')
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError('graph should be directed acyclic')
    forward_deque = deque([])
    forward_visited = set()
    backward_deque = deque(x)
    backward_visited = set()
    ancestors_or_z = set().union(*[nx.ancestors(G, node) for node in x]) | z | x
    while forward_deque or backward_deque:
        if backward_deque:
            node = backward_deque.popleft()
            backward_visited.add(node)
            if node in y:
                return False
            if node in z:
                continue
            backward_deque.extend(G.pred[node].keys() - backward_visited)
            forward_deque.extend(G.succ[node].keys() - forward_visited)
        if forward_deque:
            node = forward_deque.popleft()
            forward_visited.add(node)
            if node in y:
                return False
            if node in ancestors_or_z:
                backward_deque.extend(G.pred[node].keys() - backward_visited)
            if node not in z:
                forward_deque.extend(G.succ[node].keys() - forward_visited)
    return True