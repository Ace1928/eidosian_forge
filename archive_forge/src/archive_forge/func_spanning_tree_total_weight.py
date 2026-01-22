from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from math import isnan
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for, py_random_state
def spanning_tree_total_weight(G, weight):
    """
        Find the sum of weights of the spanning trees of `G` using the
        appropriate `method`.

        This is easy if the chosen method is 'multiplicative', since we can
        use Kirchhoff's Tree Matrix Theorem directly. However, with the
        'additive' method, this process is slightly more complex and less
        computationally efficient as we have to find the number of spanning
        trees which contain each possible edge in the graph.

        Parameters
        ----------
        G : NetworkX Graph
            The graph to find the total weight of all spanning trees on.

        weight : string
            The key for the weight edge attribute of the graph.

        Returns
        -------
        float
            The sum of either the multiplicative or additive weight for all
            spanning trees in the graph.
        """
    if multiplicative:
        return nx.total_spanning_tree_weight(G, weight)
    elif G.number_of_edges() == 1:
        return G.edges(data=weight).__iter__().__next__()[2]
    else:
        total = 0
        for u, v, w in G.edges(data=weight):
            total += w * nx.total_spanning_tree_weight(nx.contracted_edge(G, edge=(u, v), self_loops=False), None)
        return total