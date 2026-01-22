import itertools
import pytest
import networkx as nx
def make_params_from_graph(G, F):
    """Returns {N, L, H, C} from the given graph."""
    num_nodes = len(G)
    L = {u: [] for u in range(num_nodes)}
    for u, v in G.edges:
        L[u].append(v)
        L[v].append(u)
    C = nx.algorithms.coloring.equitable_coloring.make_C_from_F(F)
    N = nx.algorithms.coloring.equitable_coloring.make_N_from_L_C(L, C)
    H = nx.algorithms.coloring.equitable_coloring.make_H_from_C_N(C, N)
    return {'N': N, 'F': F, 'C': C, 'H': H, 'L': L}