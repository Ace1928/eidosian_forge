import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch(graphs=None)
def from_numpy_array(A, parallel_edges=False, create_using=None, edge_attr='weight'):
    """Returns a graph from a 2D NumPy array.

    The 2D NumPy array is interpreted as an adjacency matrix for the graph.

    Parameters
    ----------
    A : a 2D numpy.ndarray
        An adjacency matrix representation of a graph

    parallel_edges : Boolean
        If this is True, `create_using` is a multigraph, and `A` is an
        integer array, then entry *(i, j)* in the array is interpreted as the
        number of parallel edges joining vertices *i* and *j* in the graph.
        If it is False, then the entries in the array are interpreted as
        the weight of a single edge joining the vertices.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    edge_attr : String, optional (default="weight")
        The attribute to which the array values are assigned on each edge. If
        it is None, edge attributes will not be assigned.

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of A corresponds to an edge from i to j.

    If `create_using` is :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph`, `parallel_edges` is True, and the
    entries of `A` are of type :class:`int`, then this function returns a
    multigraph (of the same type as `create_using`) with parallel edges.

    If `create_using` indicates an undirected multigraph, then only the edges
    indicated by the upper triangle of the array `A` will be added to the
    graph.

    If `edge_attr` is Falsy (False or None), edge attributes will not be
    assigned, and the array data will be treated like a binary mask of
    edge presence or absence. Otherwise, the attributes will be assigned
    as follows:

    If the NumPy array has a single data type for each array entry it
    will be converted to an appropriate Python data type.

    If the NumPy array has a user-specified compound data type the names
    of the data fields will be used as attribute keys in the resulting
    NetworkX graph.

    See Also
    --------
    to_numpy_array

    Examples
    --------
    Simple integer weights on edges:

    >>> import numpy as np
    >>> A = np.array([[1, 1], [2, 1]])
    >>> G = nx.from_numpy_array(A)
    >>> G.edges(data=True)
    EdgeDataView([(0, 0, {'weight': 1}), (0, 1, {'weight': 2}), (1, 1, {'weight': 1})])

    If `create_using` indicates a multigraph and the array has only integer
    entries and `parallel_edges` is False, then the entries will be treated
    as weights for edges joining the nodes (without creating parallel edges):

    >>> A = np.array([[1, 1], [1, 2]])
    >>> G = nx.from_numpy_array(A, create_using=nx.MultiGraph)
    >>> G[1][1]
    AtlasView({0: {'weight': 2}})

    If `create_using` indicates a multigraph and the array has only integer
    entries and `parallel_edges` is True, then the entries will be treated
    as the number of parallel edges joining those two vertices:

    >>> A = np.array([[1, 1], [1, 2]])
    >>> temp = nx.MultiGraph()
    >>> G = nx.from_numpy_array(A, parallel_edges=True, create_using=temp)
    >>> G[1][1]
    AtlasView({0: {'weight': 1}, 1: {'weight': 1}})

    User defined compound data type on edges:

    >>> dt = [("weight", float), ("cost", int)]
    >>> A = np.array([[(1.0, 2)]], dtype=dt)
    >>> G = nx.from_numpy_array(A)
    >>> G.edges()
    EdgeView([(0, 0)])
    >>> G[0][0]["cost"]
    2
    >>> G[0][0]["weight"]
    1.0

    """
    kind_to_python_type = {'f': float, 'i': int, 'u': int, 'b': bool, 'c': complex, 'S': str, 'U': str, 'V': 'void'}
    G = nx.empty_graph(0, create_using)
    if A.ndim != 2:
        raise nx.NetworkXError(f'Input array must be 2D, not {A.ndim}')
    n, m = A.shape
    if n != m:
        raise nx.NetworkXError(f'Adjacency matrix not square: nx,ny={A.shape}')
    dt = A.dtype
    try:
        python_type = kind_to_python_type[dt.kind]
    except Exception as err:
        raise TypeError(f'Unknown numpy data type: {dt}') from err
    G.add_nodes_from(range(n))
    edges = ((int(e[0]), int(e[1])) for e in zip(*A.nonzero()))
    if python_type == 'void':
        fields = sorted(((offset, dtype, name) for name, (dtype, offset) in A.dtype.fields.items()))
        triples = ((u, v, {} if edge_attr in [False, None] else {name: kind_to_python_type[dtype.kind](val) for (_, dtype, name), val in zip(fields, A[u, v])}) for u, v in edges)
    elif python_type is int and G.is_multigraph() and parallel_edges:
        chain = itertools.chain.from_iterable
        if edge_attr in [False, None]:
            triples = chain((((u, v, {}) for d in range(A[u, v])) for u, v in edges))
        else:
            triples = chain((((u, v, {edge_attr: 1}) for d in range(A[u, v])) for u, v in edges))
    elif edge_attr in [False, None]:
        triples = ((u, v, {}) for u, v in edges)
    else:
        triples = ((u, v, {edge_attr: python_type(A[u, v])}) for u, v in edges)
    if G.is_multigraph() and (not G.is_directed()):
        triples = ((u, v, d) for u, v, d in triples if u <= v)
    G.add_edges_from(triples)
    return G