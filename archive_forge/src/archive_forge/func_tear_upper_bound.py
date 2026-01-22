import copy, logging
from pyomo.common.dependencies import numpy
def tear_upper_bound(self, G):
    """
        This function quickly finds a sub-optimal set of tear
        edges. This serves as an initial upperbound when looking
        for an optimal tear set. Having an initial upper bound
        improves efficiency.

        This works by constructing a search tree and just makes a
        tear set out of all the back edges.
        """

    def cyc(node, depth):
        depths[node] = depth
        depth += 1
        for edge in G.out_edges(node, keys=True):
            suc, key = (edge[1], edge[2])
            if depths[suc] is None:
                parents[suc] = node
                cyc(suc, depth)
            elif depths[suc] < depths[node]:
                tearSet.append(edge_list.index((node, suc, key)))
    tearSet = []
    edge_list = self.idx_to_edge(G)
    depths = {}
    parents = {}
    for node in G.nodes:
        depths[node] = None
        parents[node] = None
    for node in G.nodes:
        if depths[node] is None:
            cyc(node, 0)
    return tearSet