from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import MultiAdjacencyView
from networkx.classes.digraph import DiGraph
from networkx.classes.multigraph import MultiGraph
from networkx.classes.reportviews import (
from networkx.exception import NetworkXError
@cached_property
def in_edges(self):
    """A view of the in edges of the graph as G.in_edges or G.in_edges().

        in_edges(self, nbunch=None, data=False, keys=False, default=None)

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.
        data : string or bool, optional (default=False)
            The edge attribute returned in 3-tuple (u, v, ddict[data]).
            If True, return edge attribute dict in 3-tuple (u, v, ddict).
            If False, return 2-tuple (u, v).
        keys : bool, optional (default=False)
            If True, return edge keys with each edge, creating 3-tuples
            (u, v, k) or with data, 4-tuples (u, v, k, d).
        default : value, optional (default=None)
            Value used for edges that don't have the requested attribute.
            Only relevant if data is not True or False.

        Returns
        -------
        in_edges : InMultiEdgeView or InMultiEdgeDataView
            A view of edge attributes, usually it iterates over (u, v)
            or (u, v, k) or (u, v, k, d) tuples of edges, but can also be
            used for attribute lookup as `edges[u, v, k]['foo']`.

        See Also
        --------
        edges
        """
    return InMultiEdgeView(self)