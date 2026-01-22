import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely.algorithms._oriented_envelope import _oriented_envelope_min_area_vectorized
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def extract_unique_points(geometry, **kwargs):
    """Returns all distinct vertices of an input geometry as a multipoint.

    Note that only 2 dimensions of the vertices are considered when testing
    for equality.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import LineString, MultiPoint, Point, Polygon
    >>> extract_unique_points(Point(0, 0))
    <MULTIPOINT (0 0)>
    >>> extract_unique_points(LineString([(0, 0), (1, 1), (1, 1)]))
    <MULTIPOINT (0 0, 1 1)>
    >>> extract_unique_points(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    <MULTIPOINT (0 0, 1 0, 1 1, 0 1)>
    >>> extract_unique_points(MultiPoint([(0, 0), (1, 1), (0, 0)]))
    <MULTIPOINT (0 0, 1 1)>
    >>> extract_unique_points(LineString())
    <MULTIPOINT EMPTY>
    """
    return lib.extract_unique_points(geometry, **kwargs)