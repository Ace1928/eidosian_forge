import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
def match_edges(u, v, pending_g, pending_h, Ce, matched_uv=None):
    """
        Parameters:
            u, v: matched vertices, u=None or v=None for
               deletion/insertion
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_uv: partial vertex edit path
                list of tuples (u, v) of previously matched vertex
                    mappings u<->v, u=None or v=None for
                    deletion/insertion

        Returns:
            list of (i, j): indices of edge mappings g<->h
            localCe: local CostMatrix of edge mappings
                (basically submatrix of Ce at cross of rows i, cols j)
        """
    M = len(pending_g)
    N = len(pending_h)
    if matched_uv is None or len(matched_uv) == 0:
        g_ind = []
        h_ind = []
    else:
        g_ind = [i for i in range(M) if pending_g[i][:2] == (u, u) or any((pending_g[i][:2] in ((p, u), (u, p), (p, p)) for p, q in matched_uv))]
        h_ind = [j for j in range(N) if pending_h[j][:2] == (v, v) or any((pending_h[j][:2] in ((q, v), (v, q), (q, q)) for p, q in matched_uv))]
    m = len(g_ind)
    n = len(h_ind)
    if m or n:
        C = extract_C(Ce.C, g_ind, h_ind, M, N)
        for k, i in enumerate(g_ind):
            g = pending_g[i][:2]
            for l, j in enumerate(h_ind):
                h = pending_h[j][:2]
                if nx.is_directed(G1) or nx.is_directed(G2):
                    if any((g == (p, u) and h == (q, v) or (g == (u, p) and h == (v, q)) for p, q in matched_uv)):
                        continue
                elif any((g in ((p, u), (u, p)) and h in ((q, v), (v, q)) for p, q in matched_uv)):
                    continue
                if g == (u, u) or any((g == (p, p) for p, q in matched_uv)):
                    continue
                if h == (v, v) or any((h == (q, q) for p, q in matched_uv)):
                    continue
                C[k, l] = inf
        localCe = make_CostMatrix(C, m, n)
        ij = [(g_ind[k] if k < m else M + h_ind[l], h_ind[l] if l < n else N + g_ind[k]) for k, l in zip(localCe.lsa_row_ind, localCe.lsa_col_ind) if k < m or l < n]
    else:
        ij = []
        localCe = CostMatrix(np.empty((0, 0)), [], [], 0)
    return (ij, localCe)