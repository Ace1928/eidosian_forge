import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
def get_edit_paths(matched_uv, pending_u, pending_v, Cv, matched_gh, pending_g, pending_h, Ce, matched_cost):
    """
        Parameters:
            matched_uv: partial vertex edit path
                list of tuples (u, v) of vertex mappings u<->v,
                u=None or v=None for deletion/insertion
            pending_u, pending_v: lists of vertices not yet mapped
            Cv: CostMatrix of pending vertex mappings
            matched_gh: partial edge edit path
                list of tuples (g, h) of edge mappings g<->h,
                g=None or h=None for deletion/insertion
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_cost: cost of partial edit path

        Returns:
            sequence of (vertex_path, edge_path, cost)
                vertex_path: complete vertex edit path
                    list of tuples (u, v) of vertex mappings u<->v,
                    u=None or v=None for deletion/insertion
                edge_path: complete edge edit path
                    list of tuples (g, h) of edge mappings g<->h,
                    g=None or h=None for deletion/insertion
                cost: total cost of edit path
            NOTE: path costs are non-increasing
        """
    if prune(matched_cost + Cv.ls + Ce.ls):
        return
    if not max(len(pending_u), len(pending_v)):
        nonlocal maxcost_value
        maxcost_value = min(maxcost_value, matched_cost)
        yield (matched_uv, matched_gh, matched_cost)
    else:
        edit_ops = get_edit_ops(matched_uv, pending_u, pending_v, Cv, pending_g, pending_h, Ce, matched_cost)
        for ij, Cv_ij, xy, Ce_xy, edit_cost in edit_ops:
            i, j = ij
            if prune(matched_cost + edit_cost + Cv_ij.ls + Ce_xy.ls):
                continue
            u = pending_u.pop(i) if i < len(pending_u) else None
            v = pending_v.pop(j) if j < len(pending_v) else None
            matched_uv.append((u, v))
            for x, y in xy:
                len_g = len(pending_g)
                len_h = len(pending_h)
                matched_gh.append((pending_g[x] if x < len_g else None, pending_h[y] if y < len_h else None))
            sortedx = sorted((x for x, y in xy))
            sortedy = sorted((y for x, y in xy))
            G = [pending_g.pop(x) if x < len(pending_g) else None for x in reversed(sortedx)]
            H = [pending_h.pop(y) if y < len(pending_h) else None for y in reversed(sortedy)]
            yield from get_edit_paths(matched_uv, pending_u, pending_v, Cv_ij, matched_gh, pending_g, pending_h, Ce_xy, matched_cost + edit_cost)
            if u is not None:
                pending_u.insert(i, u)
            if v is not None:
                pending_v.insert(j, v)
            matched_uv.pop()
            for x, g in zip(sortedx, reversed(G)):
                if g is not None:
                    pending_g.insert(x, g)
            for y, h in zip(sortedy, reversed(H)):
                if h is not None:
                    pending_h.insert(y, h)
            for _ in xy:
                matched_gh.pop()