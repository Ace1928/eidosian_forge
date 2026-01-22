from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from networkx.exception import NetworkXError
def to_undirected(self, as_view=False):
    """Returns an undirected copy of the graph.

        Parameters
        ----------
        as_view : bool (optional, default=False)
          If True return a view of the original undirected graph.

        Returns
        -------
        G : Graph/MultiGraph
            A deepcopy of the graph.

        See Also
        --------
        Graph, copy, add_edge, add_edges_from

        Notes
        -----
        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.

        This is in contrast to the similar `G = nx.DiGraph(D)` which returns a
        shallow copy of the data.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Warning: If you have subclassed DiGraph to use dict-like objects
        in the data structure, those changes do not transfer to the
        Graph created by this method.

        Examples
        --------
        >>> G = nx.path_graph(2)  # or MultiGraph, etc
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1), (1, 0)]
        >>> G2 = H.to_undirected()
        >>> list(G2.edges)
        [(0, 1)]
        """
    graph_class = self.to_undirected_class()
    if as_view is True:
        return nx.graphviews.generic_graph_view(self, graph_class)
    G = graph_class()
    G.graph.update(deepcopy(self.graph))
    G.add_nodes_from(((n, deepcopy(d)) for n, d in self._node.items()))
    G.add_edges_from(((u, v, deepcopy(d)) for u, nbrs in self._adj.items() for v, d in nbrs.items()))
    return G