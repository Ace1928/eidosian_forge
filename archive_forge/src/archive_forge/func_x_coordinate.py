from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def x_coordinate(self, tet, pt):
    """
        Returns the X-coordinate for the tetrahedron with index tet
        at the point pt (quadruple of integers adding up to N).

        See Definition 10.9:
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711
        """
    result = 1
    for v0 in range(4):
        for v1 in range(v0 + 1, 4):
            e = [_kronecker_delta(v0, i) + _kronecker_delta(v1, i) for i in range(4)]
            p = [x1 - x2 for x1, x2 in zip(pt, e)]
            if all((x >= 0 for x in p)):
                result *= self._shape_at_tet_point_and_edge(tet, p, e)
    return -result