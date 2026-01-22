import warnings
import numpy as np
from shapely import lib
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def is_valid_input(geometry, **kwargs):
    """Returns True if the object is a geometry or None

    Parameters
    ----------
    geometry : any object or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_geometry : checks if an object is a geometry
    is_missing : checks if an object is None

    Examples
    --------
    >>> from shapely import GeometryCollection, Point
    >>> is_valid_input(Point(0, 0))
    True
    >>> is_valid_input(GeometryCollection())
    True
    >>> is_valid_input(None)
    True
    >>> is_valid_input(1.0)
    False
    >>> is_valid_input("text")
    False
    """
    return lib.is_valid_input(geometry, **kwargs)