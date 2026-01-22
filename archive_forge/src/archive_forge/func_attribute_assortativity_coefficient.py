import networkx as nx
from networkx.algorithms.assortativity.mixing import (
from networkx.algorithms.assortativity.pairs import node_degree_xy
@nx._dispatch(node_attrs='attribute')
def attribute_assortativity_coefficient(G, attribute, nodes=None):
    """Compute assortativity for node attributes.

    Assortativity measures the similarity of connections
    in the graph with respect to the given attribute.

    Parameters
    ----------
    G : NetworkX graph

    attribute : string
        Node attribute key

    nodes: list or iterable (optional)
        Compute attribute assortativity for nodes in container.
        The default is all nodes.

    Returns
    -------
    r: float
       Assortativity of graph for given attribute

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_nodes_from([0, 1], color="red")
    >>> G.add_nodes_from([2, 3], color="blue")
    >>> G.add_edges_from([(0, 1), (2, 3)])
    >>> print(nx.attribute_assortativity_coefficient(G, "color"))
    1.0

    Notes
    -----
    This computes Eq. (2) in Ref. [1]_ , (trace(M)-sum(M^2))/(1-sum(M^2)),
    where M is the joint probability distribution (mixing matrix)
    of the specified attribute.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    """
    M = attribute_mixing_matrix(G, attribute, nodes)
    return attribute_ac(M)