from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import NetworkXError, convert
from networkx.classes.coreviews import MultiAdjacencyView
from networkx.classes.graph import Graph
from networkx.classes.reportviews import MultiDegreeView, MultiEdgeView
def new_edge_key(self, u, v):
    """Returns an unused key for edges between nodes `u` and `v`.

        The nodes `u` and `v` do not need to be already in the graph.

        Notes
        -----
        In the standard MultiGraph class the new key is the number of existing
        edges between `u` and `v` (increased if necessary to ensure unused).
        The first edge will have key 0, then 1, etc. If an edge is removed
        further new_edge_keys may not be in this order.

        Parameters
        ----------
        u, v : nodes

        Returns
        -------
        key : int
        """
    try:
        keydict = self._adj[u][v]
    except KeyError:
        return 0
    key = len(keydict)
    while key in keydict:
        key += 1
    return key