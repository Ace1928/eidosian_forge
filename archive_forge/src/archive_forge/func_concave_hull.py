import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely.algorithms._oriented_envelope import _oriented_envelope_min_area_vectorized
from shapely.decorators import multithreading_enabled, requires_geos
@requires_geos('3.11.0')
@multithreading_enabled
def concave_hull(geometry, ratio=0.0, allow_holes=False, **kwargs):
    """Computes a concave geometry that encloses an input geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
    ratio : float, default 0.0
        Number in the range [0, 1]. Higher numbers will include fewer vertices
        in the hull.
    allow_holes : bool, default False
        If set to True, the concave hull may have holes.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import MultiPoint, Polygon
    >>> concave_hull(MultiPoint([(0, 0), (0, 3), (1, 1), (3, 0), (3, 3)]), ratio=0.1)
    <POLYGON ((0 0, 0 3, 1 1, 3 3, 3 0, 0 0))>
    >>> concave_hull(MultiPoint([(0, 0), (0, 3), (1, 1), (3, 0), (3, 3)]), ratio=1.0)
    <POLYGON ((0 0, 0 3, 3 3, 3 0, 0 0))>
    >>> concave_hull(Polygon())
    <POLYGON EMPTY>
    """
    if not np.isscalar(ratio):
        raise TypeError('ratio must be scalar')
    if not np.isscalar(allow_holes):
        raise TypeError('allow_holes must be scalar')
    return lib.concave_hull(geometry, np.double(ratio), np.bool_(allow_holes), **kwargs)