import copy, logging
from pyomo.common.dependencies import numpy
def scc_calculation_order(self, sccNodes, ie, oe):
    """
        This determines the order in which to do calculations for strongly
        connected components. It is used to help determine the most efficient
        order to solve tear streams to prevent extra iterations. This just
        makes an adjacency list with the SCCs as nodes and calls the tree
        order function.

        Arguments
        ---------
            sccNodes
                List of lists of nodes in each SCC
            ie
                List of lists of in edge indexes to SCCs
            oe
                List of lists of out edge indexes to SCCs

        """
    adj = []
    adjR = []
    for i in range(len(sccNodes)):
        adj.append([])
        adjR.append([])
    done = False
    for i in range(len(sccNodes)):
        for j in range(len(sccNodes)):
            for ine in ie[i]:
                for oute in oe[j]:
                    if ine == oute:
                        adj[j].append(i)
                        adjR[i].append(j)
                        done = True
                if done:
                    break
            if done:
                break
        done = False
    return self.tree_order(adj, adjR)