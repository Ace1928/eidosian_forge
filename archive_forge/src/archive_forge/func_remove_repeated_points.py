import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely.algorithms._oriented_envelope import _oriented_envelope_min_area_vectorized
from shapely.decorators import multithreading_enabled, requires_geos
@requires_geos('3.11.0')
@multithreading_enabled
def remove_repeated_points(geometry, tolerance=0.0, **kwargs):
    """Returns a copy of a Geometry with repeated points removed.

    From the start of the coordinate sequence, each next point within the
    tolerance is removed.

    Removing repeated points with a non-zero tolerance may result in an invalid
    geometry being returned.

    Parameters
    ----------
    geometry : Geometry or array_like
    tolerance : float or array_like, default=0.0
        Use 0.0 to remove only exactly repeated points.

    Examples
    --------
    >>> from shapely import LineString, Polygon
    >>> remove_repeated_points(LineString([(0,0), (0,0), (1,0)]), tolerance=0)
    <LINESTRING (0 0, 1 0)>
    >>> remove_repeated_points(Polygon([(0, 0), (0, .5), (0, 1), (.5, 1), (0,0)]), tolerance=.5)
    <POLYGON ((0 0, 0 1, 0 0, 0 0))>
    """
    return lib.remove_repeated_points(geometry, tolerance, **kwargs)