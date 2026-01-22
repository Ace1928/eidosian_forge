import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch
def is_regular(G):
    """Determines whether the graph ``G`` is a regular graph.

    A regular graph is a graph where each vertex has the same degree. A
    regular digraph is a graph where the indegree and outdegree of each
    vertex are equal.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    bool
        Whether the given graph or digraph is regular.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 1)])
    >>> nx.is_regular(G)
    True

    """
    n1 = nx.utils.arbitrary_element(G)
    if not G.is_directed():
        d1 = G.degree(n1)
        return all((d1 == d for _, d in G.degree))
    else:
        d_in = G.in_degree(n1)
        in_regular = all((d_in == d for _, d in G.in_degree))
        d_out = G.out_degree(n1)
        out_regular = all((d_out == d for _, d in G.out_degree))
        return in_regular and out_regular