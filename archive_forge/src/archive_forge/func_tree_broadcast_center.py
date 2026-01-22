import networkx as nx
from networkx import NetworkXError
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def tree_broadcast_center(G):
    """Return the Broadcast Center of the tree `G`.

    The broadcast center of a graph G denotes the set of nodes having
    minimum broadcast time [1]_. This is a linear algorithm for determining
    the broadcast center of a tree with ``N`` nodes, as a by-product it also
    determines the broadcast time from the broadcast center.

    Parameters
    ----------
    G : undirected graph
        The graph should be an undirected tree

    Returns
    -------
    BC : (int, set) tuple
        minimum broadcast number of the tree, set of broadcast centers

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    References
    ----------
    .. [1] Slater, P.J., Cockayne, E.J., Hedetniemi, S.T,
       Information dissemination in trees. SIAM J.Comput. 10(4), 692–701 (1981)
    """
    if not nx.is_tree(G):
        NetworkXError('Input graph is not a tree')
    if G.number_of_nodes() == 2:
        return (1, set(G.nodes()))
    if G.number_of_nodes() == 1:
        return (0, set(G.nodes()))
    U = {node for node, deg in G.degree if deg == 1}
    values = {n: 0 for n in U}
    T = G.copy()
    T.remove_nodes_from(U)
    W = {node for node, deg in T.degree if deg == 1}
    values.update(((w, G.degree[w] - 1) for w in W))
    while T.number_of_nodes() >= 2:
        w = min(W, key=lambda n: values[n])
        v = next(T.neighbors(w))
        U.add(w)
        W.remove(w)
        T.remove_node(w)
        if T.degree(v) == 1:
            values.update({v: _get_max_broadcast_value(G, U, v, values)})
            W.add(v)
    v = nx.utils.arbitrary_element(T)
    b_T = _get_max_broadcast_value(G, U, v, values)
    return (b_T, _get_broadcast_centers(G, v, values, b_T))