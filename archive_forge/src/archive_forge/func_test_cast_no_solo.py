import itertools
import pytest
import networkx as nx
def test_cast_no_solo(self):
    L = {0: [8, 9], 1: [10, 11], 2: [8], 3: [9], 4: [10, 11], 5: [8], 6: [9], 7: [10, 11], 8: [0, 2, 5], 9: [0, 3, 6], 10: [1, 4, 7], 11: [1, 4, 7]}
    F = {0: 0, 1: 0, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 1, 9: 1, 10: 1, 11: 1}
    C = nx.algorithms.coloring.equitable_coloring.make_C_from_F(F)
    N = nx.algorithms.coloring.equitable_coloring.make_N_from_L_C(L, C)
    H = nx.algorithms.coloring.equitable_coloring.make_H_from_C_N(C, N)
    nx.algorithms.coloring.equitable_coloring.procedure_P(V_minus=0, V_plus=1, N=N, H=H, F=F, C=C, L=L)
    check_state(L=L, N=N, H=H, F=F, C=C)