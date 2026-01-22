import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
@nx._dispatch(graphs='R', preserve_edge_attrs={'R': {'capacity': float('inf'), 'flow': 0}}, preserve_graph_attrs=True)
def edmonds_karp_core(R, s, t, cutoff):
    """Implementation of the Edmonds-Karp algorithm."""
    R_nodes = R.nodes
    R_pred = R.pred
    R_succ = R.succ
    inf = R.graph['inf']

    def augment(path):
        """Augment flow along a path from s to t."""
        flow = inf
        it = iter(path)
        u = next(it)
        for v in it:
            attr = R_succ[u][v]
            flow = min(flow, attr['capacity'] - attr['flow'])
            u = v
        if flow * 2 > inf:
            raise nx.NetworkXUnbounded('Infinite capacity path, flow unbounded above.')
        it = iter(path)
        u = next(it)
        for v in it:
            R_succ[u][v]['flow'] += flow
            R_succ[v][u]['flow'] -= flow
            u = v
        return flow

    def bidirectional_bfs():
        """Bidirectional breadth-first search for an augmenting path."""
        pred = {s: None}
        q_s = [s]
        succ = {t: None}
        q_t = [t]
        while True:
            q = []
            if len(q_s) <= len(q_t):
                for u in q_s:
                    for v, attr in R_succ[u].items():
                        if v not in pred and attr['flow'] < attr['capacity']:
                            pred[v] = u
                            if v in succ:
                                return (v, pred, succ)
                            q.append(v)
                if not q:
                    return (None, None, None)
                q_s = q
            else:
                for u in q_t:
                    for v, attr in R_pred[u].items():
                        if v not in succ and attr['flow'] < attr['capacity']:
                            succ[v] = u
                            if v in pred:
                                return (v, pred, succ)
                            q.append(v)
                if not q:
                    return (None, None, None)
                q_t = q
    flow_value = 0
    while flow_value < cutoff:
        v, pred, succ = bidirectional_bfs()
        if pred is None:
            break
        path = [v]
        u = v
        while u != s:
            u = pred[u]
            path.append(u)
        path.reverse()
        u = v
        while u != t:
            u = succ[u]
            path.append(u)
        flow_value += augment(path)
    return flow_value