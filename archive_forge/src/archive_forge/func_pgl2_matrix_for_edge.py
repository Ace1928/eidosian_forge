from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def pgl2_matrix_for_edge(self, e):
    """
        e is a TruncatedComplex.Edge
        """
    RF = self.vertex_gram_matrices[0].base_ring()
    CF = RF.complex_field()
    if e.subcomplex_type == 'alpha':
        tet_index, perm = e.tet_and_perm
        v = self.vertex_gram_matrices[tet_index][perm[0], perm[1]]
        return matrix(CF, [[0, (v ** 2 - 1).sqrt() - v], [1, 0]])
    if e.subcomplex_type == 'beta':
        etaHalf = self._compute_eta(e.tet_and_perm) / 2
        c = etaHalf.cos()
        s = etaHalf.sin()
        return matrix(CF, [[-c, s], [s, c]])
    if e.subcomplex_type == 'gamma':
        theta = self._compute_theta(e.tet_and_perm)
        return matrix(CF, [[(theta * sage.all.I).exp(), 0], [0, 1]])
    raise Exception('Unsupported edge type')