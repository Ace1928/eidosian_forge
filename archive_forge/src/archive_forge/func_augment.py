from collections import deque
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
def augment(u, v):
    """Augmentation stage.

        Reconstruct path and determine its residual capacity.
        We start from a connecting edge, which links a node
        from the source tree to a node from the target tree.
        The connecting edge is the output of the grow function
        and the input of this function.
        """
    attr = R_succ[u][v]
    flow = min(INF, attr['capacity'] - attr['flow'])
    path = [u]
    w = u
    while w != s:
        n = w
        w = source_tree[n]
        attr = R_pred[n][w]
        flow = min(flow, attr['capacity'] - attr['flow'])
        path.append(w)
    path.reverse()
    path.append(v)
    w = v
    while w != t:
        n = w
        w = target_tree[n]
        attr = R_succ[n][w]
        flow = min(flow, attr['capacity'] - attr['flow'])
        path.append(w)
    it = iter(path)
    u = next(it)
    these_orphans = []
    for v in it:
        R_succ[u][v]['flow'] += flow
        R_succ[v][u]['flow'] -= flow
        if R_succ[u][v]['flow'] == R_succ[u][v]['capacity']:
            if v in source_tree:
                source_tree[v] = None
                these_orphans.append(v)
            if u in target_tree:
                target_tree[u] = None
                these_orphans.append(u)
        u = v
    orphans.extend(sorted(these_orphans, key=dist.get))
    return flow