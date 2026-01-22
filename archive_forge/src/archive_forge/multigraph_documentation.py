from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import NetworkXError, convert
from networkx.classes.coreviews import MultiAdjacencyView
from networkx.classes.graph import Graph
from networkx.classes.reportviews import MultiDegreeView, MultiEdgeView
Returns the number of edges between two nodes.

        Parameters
        ----------
        u, v : nodes, optional (Default=all edges)
            If u and v are specified, return the number of edges between
            u and v. Otherwise return the total number of all edges.

        Returns
        -------
        nedges : int
            The number of edges in the graph.  If nodes `u` and `v` are
            specified return the number of edges between those nodes. If
            the graph is directed, this only returns the number of edges
            from `u` to `v`.

        See Also
        --------
        size

        Examples
        --------
        For undirected multigraphs, this method counts the total number
        of edges in the graph::

            >>> G = nx.MultiGraph()
            >>> G.add_edges_from([(0, 1), (0, 1), (1, 2)])
            [0, 1, 0]
            >>> G.number_of_edges()
            3

        If you specify two nodes, this counts the total number of edges
        joining the two nodes::

            >>> G.number_of_edges(0, 1)
            2

        For directed multigraphs, this method can count the total number
        of directed edges from `u` to `v`::

            >>> G = nx.MultiDiGraph()
            >>> G.add_edges_from([(0, 1), (0, 1), (1, 0)])
            [0, 1, 0]
            >>> G.number_of_edges(0, 1)
            2
            >>> G.number_of_edges(1, 0)
            1

        