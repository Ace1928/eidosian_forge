import itertools
import networkx as nx
from networkx.convert_matrix import _generate_weighted_edges
@nx._dispatch(graphs=None)
def from_biadjacency_matrix(A, create_using=None, edge_attribute='weight'):
    """Creates a new bipartite graph from a biadjacency matrix given as a
    SciPy sparse array.

    Parameters
    ----------
    A: scipy sparse array
      A biadjacency matrix representation of a graph

    create_using: NetworkX graph
       Use specified graph for result.  The default is Graph()

    edge_attribute: string
       Name of edge attribute to store matrix numeric value. The data will
       have the same type as the matrix entry (int, float, (real,imag)).

    Notes
    -----
    The nodes are labeled with the attribute `bipartite` set to an integer
    0 or 1 representing membership in part 0 or part 1 of the bipartite graph.

    If `create_using` is an instance of :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph` and the entries of `A` are of
    type :class:`int`, then this function returns a multigraph (of the same
    type as `create_using`) with parallel edges. In this case, `edge_attribute`
    will be ignored.

    See Also
    --------
    biadjacency_matrix
    from_numpy_array

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Adjacency_matrix#Adjacency_matrix_of_a_bipartite_graph
    """
    G = nx.empty_graph(0, create_using)
    n, m = A.shape
    G.add_nodes_from(range(n), bipartite=0)
    G.add_nodes_from(range(n, n + m), bipartite=1)
    triples = ((u, n + v, d) for u, v, d in _generate_weighted_edges(A))
    if A.dtype.kind in ('i', 'u') and G.is_multigraph():
        chain = itertools.chain.from_iterable
        triples = chain((((u, v, 1) for d in range(w)) for u, v, w in triples))
    G.add_weighted_edges_from(triples, weight=edge_attribute)
    return G