import warnings
import numpy as np
from shapely import lib
from shapely.decorators import multithreading_enabled, requires_geos
def is_valid_reason(geometry, **kwargs):
    """Returns a string stating if a geometry is valid and if not, why.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted. Returns None for missing values.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_valid : returns True or False

    Examples
    --------
    >>> from shapely import LineString, Polygon
    >>> is_valid_reason(LineString([(0, 0), (1, 1)]))
    'Valid Geometry'
    >>> is_valid_reason(Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)]))
    'Ring Self-intersection[1 1]'
    >>> is_valid_reason(None) is None
    True
    """
    return lib.is_valid_reason(geometry, **kwargs)