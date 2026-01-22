import networkx as nx
from networkx.exception import NetworkXAlgorithmError
from networkx.utils import not_implemented_for
@not_implemented_for('multigraph')
@nx._dispatch(graphs='B', preserve_all_attrs=True)
def generic_weighted_projected_graph(B, nodes, weight_function=None):
    """Weighted projection of B with a user-specified weight function.

    The bipartite network B is projected on to the specified nodes
    with weights computed by a user-specified function.  This function
    must accept as a parameter the neighborhood sets of two nodes and
    return an integer or a float.

    The nodes retain their attributes and are connected in the resulting graph
    if they have an edge to a common node in the original graph.

    Parameters
    ----------
    B : NetworkX graph
        The input graph should be bipartite.

    nodes : list or iterable
        Nodes to project onto (the "bottom" nodes).

    weight_function : function
        This function must accept as parameters the same input graph
        that this function, and two nodes; and return an integer or a float.
        The default function computes the number of shared neighbors.

    Returns
    -------
    Graph : NetworkX graph
       A graph that is the projection onto the given nodes.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> # Define some custom weight functions
    >>> def jaccard(G, u, v):
    ...     unbrs = set(G[u])
    ...     vnbrs = set(G[v])
    ...     return float(len(unbrs & vnbrs)) / len(unbrs | vnbrs)
    ...
    >>> def my_weight(G, u, v, weight="weight"):
    ...     w = 0
    ...     for nbr in set(G[u]) & set(G[v]):
    ...         w += G[u][nbr].get(weight, 1) + G[v][nbr].get(weight, 1)
    ...     return w
    ...
    >>> # A complete bipartite graph with 4 nodes and 4 edges
    >>> B = nx.complete_bipartite_graph(2, 2)
    >>> # Add some arbitrary weight to the edges
    >>> for i, (u, v) in enumerate(B.edges()):
    ...     B.edges[u, v]["weight"] = i + 1
    ...
    >>> for edge in B.edges(data=True):
    ...     print(edge)
    ...
    (0, 2, {'weight': 1})
    (0, 3, {'weight': 2})
    (1, 2, {'weight': 3})
    (1, 3, {'weight': 4})
    >>> # By default, the weight is the number of shared neighbors
    >>> G = bipartite.generic_weighted_projected_graph(B, [0, 1])
    >>> print(list(G.edges(data=True)))
    [(0, 1, {'weight': 2})]
    >>> # To specify a custom weight function use the weight_function parameter
    >>> G = bipartite.generic_weighted_projected_graph(
    ...     B, [0, 1], weight_function=jaccard
    ... )
    >>> print(list(G.edges(data=True)))
    [(0, 1, {'weight': 1.0})]
    >>> G = bipartite.generic_weighted_projected_graph(
    ...     B, [0, 1], weight_function=my_weight
    ... )
    >>> print(list(G.edges(data=True)))
    [(0, 1, {'weight': 10})]

    Notes
    -----
    No attempt is made to verify that the input graph B is bipartite.
    The graph and node properties are (shallow) copied to the projected graph.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    is_bipartite,
    is_bipartite_node_set,
    sets,
    weighted_projected_graph,
    collaboration_weighted_projected_graph,
    overlap_weighted_projected_graph,
    projected_graph

    """
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    if weight_function is None:

        def weight_function(G, u, v):
            return len(set(G[u]) & set(pred[v]))
    G.graph.update(B.graph)
    G.add_nodes_from(((n, B.nodes[n]) for n in nodes))
    for u in nodes:
        nbrs2 = {n for nbr in set(B[u]) for n in B[nbr]} - {u}
        for v in nbrs2:
            weight = weight_function(B, u, v)
            G.add_edge(u, v, weight=weight)
    return G