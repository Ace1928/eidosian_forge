import snappy
import FXrays

    For a one-cusped manifold, returns all the nonempty boundary slopes of
    spun normal surfaces.  Provided the triangulation supports a
    genuine hyperbolic structure, then by `Thurston and Walsh
    <http://arxiv.org/abs/math/0503027>`_ any strict boundary slope
    (the boundary of an essential surface which is not a fiber or
    semifiber) must be listed here.

    >>> M = Manifold('K3_1')
    >>> M.normal_boundary_slopes()
    [(16, -1), (20, -1), (37, -2)]

    If the ``subset`` flag is set to ``'kabaya'``, then it only
    returns boundary slopes associated to vertex surfaces with a quad
    in every tetrahedron; by Theorem 1.1. of `[DG]
    <http://arxiv.org/abs/1102.4588>`_ these are all strict boundary
    slopes.

    >>> N = Manifold('m113')
    >>> N.normal_boundary_slopes()
    [(1, 1), (1, 2), (2, -1), (2, 3), (8, 11)]
    >>> N.normal_boundary_slopes('kabaya')
    [(8, 11)]

    If the ``subset`` flag is set to ``'brasile'`` then it returns
    only the boundary slopes that are associated to vertex surfaces
    giving isolated rays in the space of embedded normal surfaces.

    >>> N.normal_boundary_slopes('brasile')
    [(1, 2), (8, 11)]
    