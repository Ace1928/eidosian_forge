import networkx as nx
@nx._dispatch(edge_attrs='weight')
def floyd_warshall_numpy(G, nodelist=None, weight='weight'):
    """Find all-pairs shortest path lengths using Floyd's algorithm.

    This algorithm for finding shortest paths takes advantage of
    matrix representations of a graph and works well for dense
    graphs where all-pairs shortest path lengths are desired.
    The results are returned as a NumPy array, distance[i, j],
    where i and j are the indexes of two nodes in nodelist.
    The entry distance[i, j] is the distance along a shortest
    path from i to j. If no path exists the distance is Inf.

    Parameters
    ----------
    G : NetworkX graph

    nodelist : list, optional (default=G.nodes)
       The rows and columns are ordered by the nodes in nodelist.
       If nodelist is None then the ordering is produced by G.nodes.
       Nodelist should include all nodes in G.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.

    Returns
    -------
    distance : 2D numpy.ndarray
        A numpy array of shortest path distances between nodes.
        If there is no path between two nodes the value is Inf.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from([(0, 1, 5), (1, 2, 2), (2, 3, -3), (1, 3, 10), (3, 2, 8)])
    >>> nx.floyd_warshall_numpy(G)
    array([[ 0.,  5.,  7.,  4.],
           [inf,  0.,  2., -1.],
           [inf, inf,  0., -3.],
           [inf, inf,  8.,  0.]])

    Notes
    -----
    Floyd's algorithm is appropriate for finding shortest paths in
    dense graphs or graphs with negative weights when Dijkstra's
    algorithm fails. This algorithm can still fail if there are negative
    cycles. It has running time $O(n^3)$ with running space of $O(n^2)$.

    Raises
    ------
    NetworkXError
        If nodelist is not a list of the nodes in G.
    """
    import numpy as np
    if nodelist is not None:
        if not len(nodelist) == len(G) == len(set(nodelist)):
            raise nx.NetworkXError('nodelist must contain every node in G with no repeats.If you wanted a subgraph of G use G.subgraph(nodelist)')
    A = nx.to_numpy_array(G, nodelist, multigraph_weight=min, weight=weight, nonedge=np.inf)
    n, m = A.shape
    np.fill_diagonal(A, 0)
    for i in range(n):
        A = np.minimum(A, A[i, :][np.newaxis, :] + A[:, i][:, np.newaxis])
    return A