import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
def get_edit_ops(matched_uv, pending_u, pending_v, Cv, pending_g, pending_h, Ce, matched_cost):
    """
        Parameters:
            matched_uv: partial vertex edit path
                list of tuples (u, v) of vertex mappings u<->v,
                u=None or v=None for deletion/insertion
            pending_u, pending_v: lists of vertices not yet mapped
            Cv: CostMatrix of pending vertex mappings
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_cost: cost of partial edit path

        Returns:
            sequence of
                (i, j): indices of vertex mapping u<->v
                Cv_ij: reduced CostMatrix of pending vertex mappings
                    (basically Cv with row i, col j removed)
                list of (x, y): indices of edge mappings g<->h
                Ce_xy: reduced CostMatrix of pending edge mappings
                    (basically Ce with rows x, cols y removed)
                cost: total cost of edit operation
            NOTE: most promising ops first
        """
    m = len(pending_u)
    n = len(pending_v)
    i, j = min(((k, l) for k, l in zip(Cv.lsa_row_ind, Cv.lsa_col_ind) if k < m or l < n))
    xy, localCe = match_edges(pending_u[i] if i < m else None, pending_v[j] if j < n else None, pending_g, pending_h, Ce, matched_uv)
    Ce_xy = reduce_Ce(Ce, xy, len(pending_g), len(pending_h))
    if prune(matched_cost + Cv.ls + localCe.ls + Ce_xy.ls):
        pass
    else:
        Cv_ij = CostMatrix(reduce_C(Cv.C, (i,), (j,), m, n), reduce_ind(Cv.lsa_row_ind, (i, m + j)), reduce_ind(Cv.lsa_col_ind, (j, n + i)), Cv.ls - Cv.C[i, j])
        yield ((i, j), Cv_ij, xy, Ce_xy, Cv.C[i, j] + localCe.ls)
    other = []
    fixed_i, fixed_j = (i, j)
    if m <= n:
        candidates = ((t, fixed_j) for t in range(m + n) if t != fixed_i and (t < m or t == m + fixed_j))
    else:
        candidates = ((fixed_i, t) for t in range(m + n) if t != fixed_j and (t < n or t == n + fixed_i))
    for i, j in candidates:
        if prune(matched_cost + Cv.C[i, j] + Ce.ls):
            continue
        Cv_ij = make_CostMatrix(reduce_C(Cv.C, (i,), (j,), m, n), m - 1 if i < m else m, n - 1 if j < n else n)
        if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + Ce.ls):
            continue
        xy, localCe = match_edges(pending_u[i] if i < m else None, pending_v[j] if j < n else None, pending_g, pending_h, Ce, matched_uv)
        if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + localCe.ls):
            continue
        Ce_xy = reduce_Ce(Ce, xy, len(pending_g), len(pending_h))
        if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + localCe.ls + Ce_xy.ls):
            continue
        other.append(((i, j), Cv_ij, xy, Ce_xy, Cv.C[i, j] + localCe.ls))
    yield from sorted(other, key=lambda t: t[4] + t[1].ls + t[3].ls)