important in operations research and theoretical computer science.
import math
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.utils import not_implemented_for, pairwise, py_random_state
def k_pi():
    """
        Find the set of minimum 1-Arborescences for G at point pi.

        Returns
        -------
        Set
            The set of minimum 1-Arborescences
        """
    G_1 = G.copy()
    minimum_1_arborescences = set()
    minimum_1_arborescence_weight = math.inf
    n = next(G.__iter__())
    G_1.remove_node(n)
    min_root = {'node': None, weight: math.inf}
    max_root = {'node': None, weight: -math.inf}
    for u, v, d in G.edges(n, data=True):
        if d[weight] < min_root[weight]:
            min_root = {'node': v, weight: d[weight]}
        if d[weight] > max_root[weight]:
            max_root = {'node': v, weight: d[weight]}
    min_in_edge = min(G.in_edges(n, data=True), key=lambda x: x[2][weight])
    min_root[weight] = min_root[weight] + min_in_edge[2][weight]
    max_root[weight] = max_root[weight] + min_in_edge[2][weight]
    min_arb_weight = math.inf
    for arb in nx.ArborescenceIterator(G_1):
        arb_weight = arb.size(weight)
        if min_arb_weight == math.inf:
            min_arb_weight = arb_weight
        elif arb_weight > min_arb_weight + max_root[weight] - min_root[weight]:
            break
        for N, deg in arb.in_degree:
            if deg == 0:
                arb.add_edge(n, N, **{weight: G[n][N][weight]})
                arb_weight += G[n][N][weight]
                break
        edge_data = G[N][n]
        G.remove_edge(N, n)
        min_weight = min(G.in_edges(n, data=weight), key=lambda x: x[2])[2]
        min_edges = [(u, v, d) for u, v, d in G.in_edges(n, data=weight) if d == min_weight]
        for u, v, d in min_edges:
            new_arb = arb.copy()
            new_arb.add_edge(u, v, **{weight: d})
            new_arb_weight = arb_weight + d
            if new_arb_weight < minimum_1_arborescence_weight:
                minimum_1_arborescences.clear()
                minimum_1_arborescence_weight = new_arb_weight
            if new_arb_weight == minimum_1_arborescence_weight:
                minimum_1_arborescences.add(new_arb)
        G.add_edge(N, n, **edge_data)
    return minimum_1_arborescences