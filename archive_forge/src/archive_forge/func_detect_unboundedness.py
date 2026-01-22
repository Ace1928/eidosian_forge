from collections import deque
import networkx as nx
@nx._dispatch(graphs='R', preserve_edge_attrs={'R': {'capacity': float('inf')}}, preserve_graph_attrs=True)
def detect_unboundedness(R, s, t):
    """Detect an infinite-capacity s-t path in R."""
    q = deque([s])
    seen = {s}
    inf = R.graph['inf']
    while q:
        u = q.popleft()
        for v, attr in R[u].items():
            if attr['capacity'] == inf and v not in seen:
                if v == t:
                    raise nx.NetworkXUnbounded('Infinite capacity path, flow unbounded above.')
                seen.add(v)
                q.append(v)