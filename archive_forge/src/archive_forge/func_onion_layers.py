import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatch
def onion_layers(G):
    """Returns the layer of each vertex in an onion decomposition of the graph.

    The onion decomposition refines the k-core decomposition by providing
    information on the internal organization of each k-shell. It is usually
    used alongside the `core numbers`.

    Parameters
    ----------
    G : NetworkX graph
        A simple graph without self loops or parallel edges

    Returns
    -------
    od_layers : dictionary
        A dictionary keyed by vertex to the onion layer. The layers are
        contiguous integers starting at 1.

    Raises
    ------
    NetworkXError
        The onion decomposition is not implemented for graphs with self loops
        or parallel edges or for directed graphs.

    Notes
    -----
    Not implemented for graphs with parallel edges or self loops.

    Not implemented for directed graphs.

    See Also
    --------
    core_number

    References
    ----------
    .. [1] Multi-scale structure and topological anomaly detection via a new
       network statistic: The onion decomposition
       L. Hébert-Dufresne, J. A. Grochow, and A. Allard
       Scientific Reports 6, 31708 (2016)
       http://doi.org/10.1038/srep31708
    .. [2] Percolation and the effective structure of complex networks
       A. Allard and L. Hébert-Dufresne
       Physical Review X 9, 011023 (2019)
       http://doi.org/10.1103/PhysRevX.9.011023
    """
    if nx.number_of_selfloops(G) > 0:
        msg = 'Input graph contains self loops which is not permitted; Consider using G.remove_edges_from(nx.selfloop_edges(G)).'
        raise NetworkXError(msg)
    od_layers = {}
    neighbors = {v: list(nx.all_neighbors(G, v)) for v in G}
    degrees = dict(G.degree())
    current_core = 1
    current_layer = 1
    isolated_nodes = list(nx.isolates(G))
    if len(isolated_nodes) > 0:
        for v in isolated_nodes:
            od_layers[v] = current_layer
            degrees.pop(v)
        current_layer = 2
    while len(degrees) > 0:
        nodes = sorted(degrees, key=degrees.get)
        min_degree = degrees[nodes[0]]
        if min_degree > current_core:
            current_core = min_degree
        this_layer = []
        for n in nodes:
            if degrees[n] > current_core:
                break
            this_layer.append(n)
        for v in this_layer:
            od_layers[v] = current_layer
            for n in neighbors[v]:
                neighbors[n].remove(v)
                degrees[n] = degrees[n] - 1
            degrees.pop(v)
        current_layer = current_layer + 1
    return od_layers