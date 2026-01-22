import networkx as nx
from networkx.utils import not_implemented_for
from .distance_measures import diameter
@not_implemented_for('directed', 'multigraph')
@nx._dispatch
def intersection_array(G):
    """Returns the intersection array of a distance-regular graph.

    Given a distance-regular graph G with integers b_i, c_i,i = 0,....,d
    such that for any 2 vertices x,y in G at a distance i=d(x,y), there
    are exactly c_i neighbors of y at a distance of i-1 from x and b_i
    neighbors of y at a distance of i+1 from x.

    A distance regular graph's intersection array is given by,
    [b_0,b_1,.....b_{d-1};c_1,c_2,.....c_d]

    Parameters
    ----------
    G: Networkx graph (undirected)

    Returns
    -------
    b,c: tuple of lists

    Examples
    --------
    >>> G = nx.icosahedral_graph()
    >>> nx.intersection_array(G)
    ([5, 2, 1], [1, 2, 5])

    References
    ----------
    .. [1] Weisstein, Eric W. "Intersection Array."
       From MathWorld--A Wolfram Web Resource.
       http://mathworld.wolfram.com/IntersectionArray.html

    See Also
    --------
    global_parameters
    """
    degree = iter(G.degree())
    _, k = next(degree)
    for _, knext in degree:
        if knext != k:
            raise nx.NetworkXError('Graph is not distance regular.')
        k = knext
    path_length = dict(nx.all_pairs_shortest_path_length(G))
    diameter = max((max(path_length[n].values()) for n in path_length))
    bint = {}
    cint = {}
    for u in G:
        for v in G:
            try:
                i = path_length[u][v]
            except KeyError as err:
                raise nx.NetworkXError('Graph is not distance regular.') from err
            c = len([n for n in G[v] if path_length[n][u] == i - 1])
            b = len([n for n in G[v] if path_length[n][u] == i + 1])
            if cint.get(i, c) != c or bint.get(i, b) != b:
                raise nx.NetworkXError('Graph is not distance regular')
            bint[i] = b
            cint[i] = c
    return ([bint.get(j, 0) for j in range(diameter)], [cint.get(j + 1, 0) for j in range(diameter)])