from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def short_edge(self, tet, v0, v1, v2):
    """
        The matrix that labels a long edge starting at vertex (v0, v1, v2)
        of a doubly truncated simplex corresponding to the ideal tetrahedron
        with index tet.

        This matrix was labeled gamma^{v0v1v2} in Figure 18 of
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711

        It is computed using equation 10.22.

        The resulting matrix is given as a python list of lists.
        """
    key = 'short_%d_%d%d%d' % (tet, v0, v1, v2)
    if key not in self._edge_cache:
        edge = [_kronecker_delta(v0, i) + _kronecker_delta(v1, i) for i in range(4)]
        sgn = CrossRatios._cyclic_three_perm_sign(v0, v1, v2)
        N = self.N()
        m = self._get_identity_matrix()
        for a0 in range(N - 1):
            a1 = N - 2 - a0
            pt = [a0 * _kronecker_delta(v0, i) + a1 * _kronecker_delta(v1, i) for i in range(4)]
            cross_ratio = self._shape_at_tet_point_and_edge(tet, pt, edge)
            m = matrix.matrix_mult(m, _H(N, a0 + 1, cross_ratio ** sgn))
        self._edge_cache[key] = m
    return self._edge_cache[key]