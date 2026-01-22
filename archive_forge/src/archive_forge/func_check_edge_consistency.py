from ..snap.t3mlite import simplex
from ..hyperboloid import *
def check_edge_consistency(m):
    RF = m.Tetrahedra[0].O13_matrices[simplex.F0].base_ring()
    id_matrix = matrix.identity(ring=RF, n=4)
    for e in m.Edges:
        t = id_matrix
        for tet, perm in e.embeddings():
            t = tet.O13_matrices[perm.image(simplex.F2)] * t
        t = t - id_matrix
        for i in range(4):
            for j in range(4):
                if abs(t[i, j]) > RF(1e-10):
                    raise Exception('Edge not gluing up')