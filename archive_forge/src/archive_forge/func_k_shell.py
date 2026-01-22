import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
@nx._dispatch(preserve_all_attrs=True)
def k_shell(G, k=None, core_number=None):
    """Returns the k-shell of G.

    The k-shell is the subgraph induced by nodes with core number k.
    That is, nodes in the k-core that are not in the (k+1)-core.

    Parameters
    ----------
    G : NetworkX graph
      A graph or directed graph.
    k : int, optional
      The order of the shell. If not specified return the outer shell.
    core_number : dictionary, optional
      Precomputed core numbers for the graph G.


    Returns
    -------
    G : NetworkX graph
       The k-shell subgraph

    Raises
    ------
    NetworkXError
        The k-shell is not implemented for graphs with self loops
        or parallel edges.

    Notes
    -----
    This is similar to k_corona but in that case only neighbors in the
    k-core are considered.

    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Graph, node, and edge attributes are copied to the subgraph.

    See Also
    --------
    core_number
    k_corona


    References
    ----------
    .. [1] A model of Internet topology using k-shell decomposition
       Shai Carmi, Shlomo Havlin, Scott Kirkpatrick, Yuval Shavitt,
       and Eran Shir, PNAS  July 3, 2007   vol. 104  no. 27  11150-11154
       http://www.pnas.org/content/104/27/11150.full
    """

    def k_filter(v, k, c):
        return c[v] == k
    return _core_subgraph(G, k_filter, k, core_number)