import numpy as np
from shapely import GeometryType, lib
from shapely.decorators import multithreading_enabled, requires_geos
from shapely.errors import UnsupportedGEOSVersionError
@requires_geos('3.8.0')
@multithreading_enabled
def coverage_union(a, b, **kwargs):
    """Merges multiple polygons into one. This is an optimized version of
    union which assumes the polygons to be non-overlapping.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    coverage_union_all

    Examples
    --------
    >>> from shapely import normalize, Polygon
    >>> polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    >>> normalize(coverage_union(polygon, Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)])))
    <POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>

    Union with None returns same polygon
    >>> normalize(coverage_union(polygon, None))
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    """
    return coverage_union_all([a, b], **kwargs)