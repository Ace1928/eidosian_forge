import itertools
import pytest
import networkx as nx
def test_case_V_plus_not_in_A_cal(self):
    L = {0: [2, 5], 1: [3, 4], 2: [0, 8], 3: [1, 7], 4: [1, 6], 5: [0, 6], 6: [4, 5], 7: [3], 8: [2]}
    F = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}
    C = nx.algorithms.coloring.equitable_coloring.make_C_from_F(F)
    N = nx.algorithms.coloring.equitable_coloring.make_N_from_L_C(L, C)
    H = nx.algorithms.coloring.equitable_coloring.make_H_from_C_N(C, N)
    nx.algorithms.coloring.equitable_coloring.procedure_P(V_minus=0, V_plus=1, N=N, H=H, F=F, C=C, L=L)
    check_state(L=L, N=N, H=H, F=F, C=C)