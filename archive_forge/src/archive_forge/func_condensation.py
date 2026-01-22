import networkx as nx
from networkx.utils.decorators import not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def condensation(G, scc=None):
    """Returns the condensation of G.

    The condensation of G is the graph with each of the strongly connected
    components contracted into a single node.

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph.

    scc:  list or generator (optional, default=None)
       Strongly connected components. If provided, the elements in
       `scc` must partition the nodes in `G`. If not provided, it will be
       calculated as scc=nx.strongly_connected_components(G).

    Returns
    -------
    C : NetworkX DiGraph
       The condensation graph C of G.  The node labels are integers
       corresponding to the index of the component in the list of
       strongly connected components of G.  C has a graph attribute named
       'mapping' with a dictionary mapping the original nodes to the
       nodes in C to which they belong.  Each node in C also has a node
       attribute 'members' with the set of original nodes in G that
       form the SCC that the node in C represents.

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    Contracting two sets of strongly connected nodes into two distinct SCC
    using the barbell graph.

    >>> G = nx.barbell_graph(4, 0)
    >>> G.remove_edge(3, 4)
    >>> G = nx.DiGraph(G)
    >>> H = nx.condensation(G)
    >>> H.nodes.data()
    NodeDataView({0: {'members': {0, 1, 2, 3}}, 1: {'members': {4, 5, 6, 7}}})
    >>> H.graph['mapping']
    {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}

    Contracting a complete graph into one single SCC.

    >>> G = nx.complete_graph(7, create_using=nx.DiGraph)
    >>> H = nx.condensation(G)
    >>> H.nodes
    NodeView((0,))
    >>> H.nodes.data()
    NodeDataView({0: {'members': {0, 1, 2, 3, 4, 5, 6}}})

    Notes
    -----
    After contracting all strongly connected components to a single node,
    the resulting graph is a directed acyclic graph.

    """
    if scc is None:
        scc = nx.strongly_connected_components(G)
    mapping = {}
    members = {}
    C = nx.DiGraph()
    C.graph['mapping'] = mapping
    if len(G) == 0:
        return C
    for i, component in enumerate(scc):
        members[i] = component
        mapping.update(((n, i) for n in component))
    number_of_components = i + 1
    C.add_nodes_from(range(number_of_components))
    C.add_edges_from(((mapping[u], mapping[v]) for u, v in G.edges() if mapping[u] != mapping[v]))
    nx.set_node_attributes(C, members, 'members')
    return C