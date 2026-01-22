from the graph class, we short-cut the chain by returning a
import networkx as nx
from networkx.classes.coreviews import (
from networkx.classes.filters import no_filter
from networkx.exception import NetworkXError
from networkx.utils import deprecate_positional_args, not_implemented_for
View of `G` with edge directions reversed

    `reverse_view` returns a read-only view of the input graph where
    edge directions are reversed.

    Identical to digraph.reverse(copy=False)

    Parameters
    ----------
    G : networkx.DiGraph

    Returns
    -------
    graph : networkx.DiGraph

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2)
    >>> G.add_edge(2, 3)
    >>> G.edges()
    OutEdgeView([(1, 2), (2, 3)])

    >>> view = nx.reverse_view(G)
    >>> view.edges()
    OutEdgeView([(2, 1), (3, 2)])
    