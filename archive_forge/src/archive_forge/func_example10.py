import itertools
from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import QQ, Vector3, Vector4, Matrix
from .barycentric_geometry import (BarycentricPoint,
from .mcomplex_with_link import McomplexWithLink
def example10():
    """
    >>> E = example10()
    >>> len(E.link_components())
    1
    """

    def BP(*vec):
        v = Vector4(vec)
        v = v / sum(v)
        return BarycentricPoint(*v)
    base_tri = [([0, 1, 0, 1], [(2, 1, 0, 3), (0, 3, 2, 1), (2, 1, 0, 3), (0, 1, 3, 2)]), ([1, 1, 0, 0], [(1, 0, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (0, 3, 2, 1)])]
    M = McomplexWithLink(base_tri)
    T0, T1 = M.Tetrahedra
    a0 = BP(17, 18, 0, 19)
    a1 = BP(0, 18, 17, 19)
    a2 = BP(10, 30, 11, 60)
    a3 = BP(30, 0, 11, 10)
    b0 = BP(14, 0, 15, 16)
    b1 = BP(0, 14, 15, 16)
    b3 = BP(30, 10, 11, 0)
    c0 = BP(5, 1, 2, 0)
    c1 = BP(4, 1, 10, 0)
    c2 = BP(20, 50, 71, 0)
    d0 = BP(5, 1, 0, 2)
    d1 = BP(4, 1, 0, 10)
    d2 = BP(20, 50, 0, 71)
    BA = BarycentricArc
    for p in [a0, a1, a2, a3, b0, b1, b3, c0, c1, c2, d0, d1, d2]:
        p.round(1117, force=True)
    T0.arcs = [BA(a1, c0), BA(c1, a0), BA(c2, a2), BA(a2, a3)]
    T1.arcs = [BA(b3, d1), BA(d0, b0), BA(b1, d2)]
    M._build_link()
    return M