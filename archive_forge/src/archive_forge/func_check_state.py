import itertools
import pytest
import networkx as nx
def check_state(L, N, H, F, C):
    s = len(C[0])
    num_colors = len(C.keys())
    assert all((u in L[v] for u in L for v in L[u]))
    assert all((F[u] != F[v] for u in L for v in L[u]))
    assert all((len(L[u]) < num_colors for u in L))
    assert all((len(C[x]) == s for x in C))
    assert all((H[c1, c2] >= 0 for c1 in C for c2 in C))
    assert all((N[u, F[u]] == 0 for u in F))