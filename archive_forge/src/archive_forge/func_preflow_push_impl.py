from collections import deque
from itertools import islice
import networkx as nx
from ...utils import arbitrary_element
from .utils import (
def preflow_push_impl(G, s, t, capacity, residual, global_relabel_freq, value_only):
    """Implementation of the highest-label preflow-push algorithm."""
    if s not in G:
        raise nx.NetworkXError(f'node {str(s)} not in graph')
    if t not in G:
        raise nx.NetworkXError(f'node {str(t)} not in graph')
    if s == t:
        raise nx.NetworkXError('source and sink are the same node')
    if global_relabel_freq is None:
        global_relabel_freq = 0
    if global_relabel_freq < 0:
        raise nx.NetworkXError('global_relabel_freq must be nonnegative.')
    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual
    detect_unboundedness(R, s, t)
    R_nodes = R.nodes
    R_pred = R.pred
    R_succ = R.succ
    for u in R:
        R_nodes[u]['excess'] = 0
        for e in R_succ[u].values():
            e['flow'] = 0

    def reverse_bfs(src):
        """Perform a reverse breadth-first search from src in the residual
        network.
        """
        heights = {src: 0}
        q = deque([(src, 0)])
        while q:
            u, height = q.popleft()
            height += 1
            for v, attr in R_pred[u].items():
                if v not in heights and attr['flow'] < attr['capacity']:
                    heights[v] = height
                    q.append((v, height))
        return heights
    heights = reverse_bfs(t)
    if s not in heights:
        R.graph['flow_value'] = 0
        return R
    n = len(R)
    max_height = max((heights[u] for u in heights if u != s))
    heights[s] = n
    grt = GlobalRelabelThreshold(n, R.size(), global_relabel_freq)
    for u in R:
        R_nodes[u]['height'] = heights[u] if u in heights else n + 1
        R_nodes[u]['curr_edge'] = CurrentEdge(R_succ[u])

    def push(u, v, flow):
        """Push flow units of flow from u to v."""
        R_succ[u][v]['flow'] += flow
        R_succ[v][u]['flow'] -= flow
        R_nodes[u]['excess'] -= flow
        R_nodes[v]['excess'] += flow
    for u, attr in R_succ[s].items():
        flow = attr['capacity']
        if flow > 0:
            push(s, u, flow)
    levels = [Level() for i in range(2 * n)]
    for u in R:
        if u != s and u != t:
            level = levels[R_nodes[u]['height']]
            if R_nodes[u]['excess'] > 0:
                level.active.add(u)
            else:
                level.inactive.add(u)

    def activate(v):
        """Move a node from the inactive set to the active set of its level."""
        if v != s and v != t:
            level = levels[R_nodes[v]['height']]
            if v in level.inactive:
                level.inactive.remove(v)
                level.active.add(v)

    def relabel(u):
        """Relabel a node to create an admissible edge."""
        grt.add_work(len(R_succ[u]))
        return min((R_nodes[v]['height'] for v, attr in R_succ[u].items() if attr['flow'] < attr['capacity'])) + 1

    def discharge(u, is_phase1):
        """Discharge a node until it becomes inactive or, during phase 1 (see
        below), its height reaches at least n. The node is known to have the
        largest height among active nodes.
        """
        height = R_nodes[u]['height']
        curr_edge = R_nodes[u]['curr_edge']
        next_height = height
        levels[height].active.remove(u)
        while True:
            v, attr = curr_edge.get()
            if height == R_nodes[v]['height'] + 1 and attr['flow'] < attr['capacity']:
                flow = min(R_nodes[u]['excess'], attr['capacity'] - attr['flow'])
                push(u, v, flow)
                activate(v)
                if R_nodes[u]['excess'] == 0:
                    levels[height].inactive.add(u)
                    break
            try:
                curr_edge.move_to_next()
            except StopIteration:
                height = relabel(u)
                if is_phase1 and height >= n - 1:
                    levels[height].active.add(u)
                    break
                next_height = height
        R_nodes[u]['height'] = height
        return next_height

    def gap_heuristic(height):
        """Apply the gap heuristic."""
        for level in islice(levels, height + 1, max_height + 1):
            for u in level.active:
                R_nodes[u]['height'] = n + 1
            for u in level.inactive:
                R_nodes[u]['height'] = n + 1
            levels[n + 1].active.update(level.active)
            level.active.clear()
            levels[n + 1].inactive.update(level.inactive)
            level.inactive.clear()

    def global_relabel(from_sink):
        """Apply the global relabeling heuristic."""
        src = t if from_sink else s
        heights = reverse_bfs(src)
        if not from_sink:
            del heights[t]
        max_height = max(heights.values())
        if from_sink:
            for u in R:
                if u not in heights and R_nodes[u]['height'] < n:
                    heights[u] = n + 1
        else:
            for u in heights:
                heights[u] += n
            max_height += n
        del heights[src]
        for u, new_height in heights.items():
            old_height = R_nodes[u]['height']
            if new_height != old_height:
                if u in levels[old_height].active:
                    levels[old_height].active.remove(u)
                    levels[new_height].active.add(u)
                else:
                    levels[old_height].inactive.remove(u)
                    levels[new_height].inactive.add(u)
                R_nodes[u]['height'] = new_height
        return max_height
    height = max_height
    while height > 0:
        while True:
            level = levels[height]
            if not level.active:
                height -= 1
                break
            old_height = height
            old_level = level
            u = arbitrary_element(level.active)
            height = discharge(u, True)
            if grt.is_reached():
                height = global_relabel(True)
                max_height = height
                grt.clear_work()
            elif not old_level.active and (not old_level.inactive):
                gap_heuristic(old_height)
                height = old_height - 1
                max_height = height
            else:
                max_height = max(max_height, height)
    if value_only:
        R.graph['flow_value'] = R_nodes[t]['excess']
        return R
    height = global_relabel(False)
    grt.clear_work()
    while height > n:
        while True:
            level = levels[height]
            if not level.active:
                height -= 1
                break
            u = arbitrary_element(level.active)
            height = discharge(u, False)
            if grt.is_reached():
                height = global_relabel(False)
                grt.clear_work()
    R.graph['flow_value'] = R_nodes[t]['excess']
    return R