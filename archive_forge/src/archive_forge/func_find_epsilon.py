important in operations research and theoretical computer science.
import math
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.utils import not_implemented_for, pairwise, py_random_state
def find_epsilon(k, d):
    """
        Given the direction of ascent at pi, find the maximum distance we can go
        in that direction.

        Parameters
        ----------
        k_xy : set
            The set of 1-arborescences which have the minimum rate of increase
            in the direction of ascent

        d : dict
            The direction of ascent

        Returns
        -------
        float
            The distance we can travel in direction `d`
        """
    min_epsilon = math.inf
    for e_u, e_v, e_w in G.edges(data=weight):
        if (e_u, e_v) in k.edges:
            continue
        if len(k.in_edges(e_v, data=weight)) > 1:
            raise Exception
        sub_u, sub_v, sub_w = next(k.in_edges(e_v, data=weight).__iter__())
        k.add_edge(e_u, e_v, **{weight: e_w})
        k.remove_edge(sub_u, sub_v)
        if max((d for n, d in k.in_degree())) <= 1 and len(G) == k.number_of_edges() and nx.is_weakly_connected(k):
            if d[sub_u] == d[e_u] or sub_w == e_w:
                k.remove_edge(e_u, e_v)
                k.add_edge(sub_u, sub_v, **{weight: sub_w})
                continue
            epsilon = (sub_w - e_w) / (d[e_u] - d[sub_u])
            if 0 < epsilon < min_epsilon:
                min_epsilon = epsilon
        k.remove_edge(e_u, e_v)
        k.add_edge(sub_u, sub_v, **{weight: sub_w})
    return min_epsilon