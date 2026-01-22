import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def get_coordinate_dimension(geometry, **kwargs):
    """Returns the dimensionality of the coordinates in a geometry (2 or 3).

    Returns -1 for missing geometries (``None`` values). Note that if the first Z
    coordinate equals ``nan``, this function will return ``2``.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import Point
    >>> get_coordinate_dimension(Point(0, 0))
    2
    >>> get_coordinate_dimension(Point(0, 0, 1))
    3
    >>> get_coordinate_dimension(None)
    -1
    >>> get_coordinate_dimension(Point(0, 0, float("nan")))
    2
    """
    return lib.get_coordinate_dimension(geometry, **kwargs)