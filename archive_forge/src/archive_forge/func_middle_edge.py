from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def middle_edge(self, tet, v0, v1, v2):
    """
        The matrix that labels a middle edge starting at vertex (v0, v1, v2)
        of a doubly truncated simplex corresponding to the ideal tetrahedron
        with index tet.

        This matrix was labeled beta^{v0v1v2} in Figure 18 of
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711

        It is computed using equation 10.22.

        The resulting matrix is given as a python list of lists.
        """
    key = 'middle_%d_%d%d%d' % (tet, v0, v1, v2)
    if key not in self._edge_cache:
        N = self.N()
        sgn = CrossRatios._cyclic_three_perm_sign(v0, v1, v2)
        m = self._get_identity_matrix()
        for k in range(1, N):
            prod1 = self._get_identity_matrix()
            for i in range(1, N - k + 1):
                prod1 = matrix.matrix_mult(prod1, _X(N, i, 1))
            prod2 = self._get_identity_matrix()
            for i in range(1, N - k):
                pt = [k * _kronecker_delta(v2, j) + i * _kronecker_delta(v0, j) + (N - k - i) * _kronecker_delta(v1, j) for j in range(4)]
                prod2 = matrix.matrix_mult(prod2, _H(N, i, self.x_coordinate(tet, pt) ** (-sgn)))
            m = matrix.matrix_mult(m, matrix.matrix_mult(prod1, prod2))
        dpm = [[-(-1) ** (N - i) * _kronecker_delta(i, j) for i in range(N)] for j in range(N)]
        m = matrix.matrix_mult(m, dpm)
        self._edge_cache[key] = m
    return self._edge_cache[key]