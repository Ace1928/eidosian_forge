from collections import defaultdict, deque
from itertools import chain, combinations, islice
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch
def make_clique_bipartite(G, fpos=None, create_using=None, name=None):
    """Returns the bipartite clique graph corresponding to `G`.

    In the returned bipartite graph, the "bottom" nodes are the nodes of
    `G` and the "top" nodes represent the maximal cliques of `G`.
    There is an edge from node *v* to clique *C* in the returned graph
    if and only if *v* is an element of *C*.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    fpos : bool
        If True or not None, the returned graph will have an
        additional attribute, `pos`, a dictionary mapping node to
        position in the Euclidean plane.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        A bipartite graph whose "bottom" set is the nodes of the graph
        `G`, whose "top" set is the cliques of `G`, and whose edges
        join nodes of `G` to the cliques that contain them.

        The nodes of the graph `G` have the node attribute
        'bipartite' set to 1 and the nodes representing cliques
        have the node attribute 'bipartite' set to 0, as is the
        convention for bipartite graphs in NetworkX.

    """
    B = nx.empty_graph(0, create_using)
    B.clear()
    B.add_nodes_from(G, bipartite=1)
    for i, cl in enumerate(find_cliques(G)):
        name = -i - 1
        B.add_node(name, bipartite=0)
        B.add_edges_from(((v, name) for v in cl))
    return B