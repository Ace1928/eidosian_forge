import copy, logging
from pyomo.common.dependencies import numpy
def scc_collect(self, G, excludeEdges=None):
    """
        This is an algorithm for finding strongly connected components (SCCs)
        in a graph. It is based on Tarjan. 1972 Depth-First Search and Linear
        Graph Algorithms, SIAM J. Comput. v1 no. 2 1972

        Returns
        -------
            sccNodes
                List of lists of nodes in each SCC
            sccEdges
                List of lists of edge indexes in each SCC
            sccOrder
                List of lists for order in which to calculate SCCs
            outEdges
                List of lists of edge indexes leaving the SCC
        """

    def sc(v, stk, depth, stringComps):
        ndepth[v] = depth
        back[v] = depth
        depth += 1
        stk.append(v)
        for w in adj[v]:
            if ndepth[w] == None:
                sc(w, stk, depth, stringComps)
                back[v] = min(back[w], back[v])
            elif w in stk:
                back[v] = min(back[w], back[v])
        if back[v] == ndepth[v]:
            scomp = []
            while True:
                w = stk.pop()
                scomp.append(i2n[w])
                if w == v:
                    break
            stringComps.append(scomp)
        return depth
    i2n, adj, _ = self.adj_lists(G, excludeEdges=excludeEdges)
    stk = []
    stringComps = []
    ndepth = [None] * len(i2n)
    back = [None] * len(i2n)
    for v in range(len(i2n)):
        if ndepth[v] == None:
            sc(v, stk, 0, stringComps)
    sccNodes = stringComps
    sccEdges = []
    outEdges = []
    inEdges = []
    for nset in stringComps:
        e, ie, oe = self.sub_graph_edges(G, nset)
        sccEdges.append(e)
        inEdges.append(ie)
        outEdges.append(oe)
    sccOrder = self.scc_calculation_order(sccNodes, inEdges, outEdges)
    return (sccNodes, sccEdges, sccOrder, outEdges)