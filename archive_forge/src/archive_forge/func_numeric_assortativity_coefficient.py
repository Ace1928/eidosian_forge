import networkx as nx
from networkx.algorithms.assortativity.mixing import (
from networkx.algorithms.assortativity.pairs import node_degree_xy
@nx._dispatch(node_attrs='attribute')
def numeric_assortativity_coefficient(G, attribute, nodes=None):
    """Compute assortativity for numerical node attributes.

    Assortativity measures the similarity of connections
    in the graph with respect to the given numeric attribute.

    Parameters
    ----------
    G : NetworkX graph

    attribute : string
        Node attribute key.

    nodes: list or iterable (optional)
        Compute numeric assortativity only for attributes of nodes in
        container. The default is all nodes.

    Returns
    -------
    r: float
       Assortativity of graph for given attribute

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_nodes_from([0, 1], size=2)
    >>> G.add_nodes_from([2, 3], size=3)
    >>> G.add_edges_from([(0, 1), (2, 3)])
    >>> print(nx.numeric_assortativity_coefficient(G, "size"))
    1.0

    Notes
    -----
    This computes Eq. (21) in Ref. [1]_ , which is the Pearson correlation
    coefficient of the specified (scalar valued) attribute across edges.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks
           Physical Review E, 67 026126, 2003
    """
    if nodes is None:
        nodes = G.nodes
    vals = {G.nodes[n][attribute] for n in nodes}
    mapping = {d: i for i, d in enumerate(vals)}
    M = attribute_mixing_matrix(G, attribute, nodes, mapping)
    return _numeric_ac(M, mapping)