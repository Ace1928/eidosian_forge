from the graph class, we short-cut the chain by returning a
import networkx as nx
from networkx.classes.coreviews import (
from networkx.classes.filters import no_filter
from networkx.exception import NetworkXError
from networkx.utils import deprecate_positional_args, not_implemented_for
def generic_graph_view(G, create_using=None):
    """Returns a read-only view of `G`.

    The graph `G` and its attributes are not copied but viewed through the new graph object
    of the same class as `G` (or of the class specified in `create_using`).

    Parameters
    ----------
    G : graph
        A directed/undirected graph/multigraph.

    create_using : NetworkX graph constructor, optional (default=None)
       Graph type to create. If graph instance, then cleared before populated.
       If `None`, then the appropriate Graph type is inferred from `G`.

    Returns
    -------
    newG : graph
        A view of the input graph `G` and its attributes as viewed through
        the `create_using` class.

    Raises
    ------
    NetworkXError
        If `G` is a multigraph (or multidigraph) but `create_using` is not, or vice versa.

    Notes
    -----
    The returned graph view is read-only (cannot modify the graph).
    Yet the view reflects any changes in `G`. The intent is to mimic dict views.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=0.3)
    >>> G.add_edge(2, 3, weight=0.5)
    >>> G.edges(data=True)
    EdgeDataView([(1, 2, {'weight': 0.3}), (2, 3, {'weight': 0.5})])

    The view exposes the attributes from the original graph.

    >>> viewG = nx.graphviews.generic_graph_view(G)
    >>> viewG.edges(data=True)
    EdgeDataView([(1, 2, {'weight': 0.3}), (2, 3, {'weight': 0.5})])

    Changes to `G` are reflected in `viewG`.

    >>> G.remove_edge(2, 3)
    >>> G.edges(data=True)
    EdgeDataView([(1, 2, {'weight': 0.3})])

    >>> viewG.edges(data=True)
    EdgeDataView([(1, 2, {'weight': 0.3})])

    We can change the graph type with the `create_using` parameter.

    >>> type(G)
    <class 'networkx.classes.graph.Graph'>
    >>> viewDG = nx.graphviews.generic_graph_view(G, create_using=nx.DiGraph)
    >>> type(viewDG)
    <class 'networkx.classes.digraph.DiGraph'>
    """
    if create_using is None:
        newG = G.__class__()
    else:
        newG = nx.empty_graph(0, create_using)
    if G.is_multigraph() != newG.is_multigraph():
        raise NetworkXError('Multigraph for G must agree with create_using')
    newG = nx.freeze(newG)
    newG._graph = G
    newG.graph = G.graph
    newG._node = G._node
    if newG.is_directed():
        if G.is_directed():
            newG._succ = G._succ
            newG._pred = G._pred
        else:
            newG._succ = G._adj
            newG._pred = G._adj
    elif G.is_directed():
        if G.is_multigraph():
            newG._adj = UnionMultiAdjacency(G._succ, G._pred)
        else:
            newG._adj = UnionAdjacency(G._succ, G._pred)
    else:
        newG._adj = G._adj
    return newG