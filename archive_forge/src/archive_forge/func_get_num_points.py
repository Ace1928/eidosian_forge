import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def get_num_points(geometry, **kwargs):
    """Returns number of points in a linestring or linearring.

    Returns 0 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of points in geometries other than linestring or linearring
        equals zero.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    get_point
    get_num_geometries

    Examples
    --------
    >>> from shapely import LineString, MultiPoint
    >>> get_num_points(LineString([(0, 0), (1, 1), (2, 2), (3, 3)]))
    4
    >>> get_num_points(MultiPoint([(0, 0), (1, 1), (2, 2), (3, 3)]))
    0
    >>> get_num_points(None)
    0
    """
    return lib.get_num_points(geometry, **kwargs)