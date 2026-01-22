import snappy
import FXrays
def normal_boundary_slopes(self, subset='all', algorithm='FXrays'):
    """
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
    """
    if not self.is_orientable():
        raise ValueError('Manifold must be orientable')
    if self.num_cusps() != 1:
        raise ValueError('More than 1 cusp, so need to look at the surfaces directly.')
    surfaces = self.normal_surfaces(algorithm)
    if subset == 'kabaya':
        surfaces = [S for S in surfaces if min(S.coefficients()) > 0]
    elif subset == 'brasile':
        isolated_surfaces = []
        for S in surfaces:
            isolated = True
            for F in surfaces:
                if S != F and S.is_compatible(F):
                    isolated = False
                    break
            if isolated:
                isolated_surfaces.append(S)
        surfaces = isolated_surfaces
    elif subset != 'all':
        raise ValueError("Subset must be one of 'all', 'kabaya', or 'brasile'")
    slopes = set([normalize_slope(S.boundary_slopes()) for S in surfaces])
    slopes.discard((0, 0))
    return sorted(slopes)