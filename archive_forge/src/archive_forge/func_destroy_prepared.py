import numpy as np
from shapely import Geometry, GeometryType, lib
from shapely._geometry_helpers import collections_1d, simple_geometries_1d
from shapely.decorators import multithreading_enabled
from shapely.io import from_wkt
def destroy_prepared(geometry, **kwargs):
    """Destroy the prepared part of a geometry, freeing up memory.

    Note that the prepared geometry will always be cleaned up if the geometry itself
    is dereferenced. This function needs only be called in very specific circumstances,
    such as freeing up memory without losing the geometries, or benchmarking.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometries are changed inplace
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    prepare
    """
    lib.destroy_prepared(geometry, **kwargs)