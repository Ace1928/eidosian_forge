from collections import deque
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
from networkx.utils import pairwise
def breath_first_search():
    parents = {}
    queue = deque([s])
    while queue:
        if t in parents:
            break
        u = queue.popleft()
        for v in R_succ[u]:
            attr = R_succ[u][v]
            if v not in parents and attr['capacity'] - attr['flow'] > 0:
                parents[v] = u
                queue.append(v)
    return parents