import copy, logging
from pyomo.common.dependencies import numpy
def tree_order(self, adj, adjR, roots=None):
    """
        This function determines the ordering of nodes in a directed
        tree. This is a generic function that can operate on any
        given tree represented by the adjaceny and reverse
        adjacency lists. If the adjacency list does not represent
        a tree the results are not valid.

        In the returned order, it is sometimes possible for more
        than one node to be calculated at once. So a list of lists
        is returned by this function. These represent a bredth
        first search order of the tree. Following the order, all
        nodes that lead to a particular node will be visited
        before it.

        Arguments
        ---------
            adj
                An adjeceny list for a directed tree. This uses
                generic integer node indexes, not node names from the
                graph itself. This allows this to be used on sub-graphs
                and graps of components more easily.
            adjR
                The reverse adjacency list coresponing to adj
            roots
                List of node indexes to start from. These do not
                need to be the root nodes of the tree, in some cases
                like when a node changes the changes may only affect
                nodes reachable in the tree from the changed node, in
                the case that roots are supplied not all the nodes in
                the tree may appear in the ordering. If no roots are
                supplied, the roots of the tree are used.
        """
    adjR = copy.deepcopy(adjR)
    for i, l in enumerate(adjR):
        adjR[i] = set(l)
    if roots is None:
        roots = []
        mark = [True] * len(adj)
        r = [True] * len(adj)
        for sucs in adj:
            for i in sucs:
                r[i] = False
        for i in range(len(r)):
            if r[i]:
                roots.append(i)
    else:
        mark = [False] * len(adj)
        lst = roots
        while len(lst) > 0:
            lst2 = []
            for i in lst:
                mark[i] = True
                lst2 += adj[i]
            lst = set(lst2)
    ndepth = [None] * len(adj)
    lst = copy.deepcopy(roots)
    order = []
    checknodes = set()
    for i in roots:
        checknodes.update(adj[i])
    depth = 0
    while len(lst) > 0:
        order.append(lst)
        depth += 1
        lst = []
        delSet = set()
        checkUpdate = set()
        for i in checknodes:
            if ndepth[i] != None:
                raise RuntimeError('Function tree_order does not work with cycles')
            remSet = set()
            for j in adjR[i]:
                if j in order[depth - 1]:
                    remSet.add(j)
                elif mark[j] == False:
                    remSet.add(j)
            adjR[i] = adjR[i].difference(remSet)
            if len(adjR[i]) == 0:
                ndepth[i] = depth
                lst.append(i)
                delSet.add(i)
                checkUpdate.update(adj[i])
        checknodes = checknodes.difference(delSet)
        checknodes = checknodes.union(checkUpdate)
    return order