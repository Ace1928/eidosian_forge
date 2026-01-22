import one of the named maximum matching algorithms directly.
import collections
import itertools
import networkx as nx
from networkx.algorithms.bipartite import sets as bipartite_sets
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
def breadth_first_search():
    for v in left:
        if leftmatches[v] is None:
            distances[v] = 0
            queue.append(v)
        else:
            distances[v] = INFINITY
    distances[None] = INFINITY
    while queue:
        v = queue.popleft()
        if distances[v] < distances[None]:
            for u in G[v]:
                if distances[rightmatches[u]] is INFINITY:
                    distances[rightmatches[u]] = distances[v] + 1
                    queue.append(rightmatches[u])
    return distances[None] is not INFINITY