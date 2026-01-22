from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
def topo_sort(relabeled):
    """Topologically sort nodes relabeled in the previous round and detect
        negative cycles.
        """
    to_scan = []
    neg_count = {}
    for u in relabeled:
        if u in neg_count:
            continue
        d_u = d[u]
        if all((d_u + weight(u, v, e) >= d[v] for v, e in G_succ[u].items())):
            continue
        stack = [(u, iter(G_succ[u].items()))]
        in_stack = {u}
        neg_count[u] = 0
        while stack:
            u, it = stack[-1]
            try:
                v, e = next(it)
            except StopIteration:
                to_scan.append(u)
                stack.pop()
                in_stack.remove(u)
                continue
            t = d[u] + weight(u, v, e)
            d_v = d[v]
            if t < d_v:
                is_neg = t < d_v
                d[v] = t
                pred[v] = u
                if v not in neg_count:
                    neg_count[v] = neg_count[u] + int(is_neg)
                    stack.append((v, iter(G_succ[v].items())))
                    in_stack.add(v)
                elif v in in_stack and neg_count[u] + int(is_neg) > neg_count[v]:
                    raise nx.NetworkXUnbounded('Negative cycle detected.')
    to_scan.reverse()
    return to_scan