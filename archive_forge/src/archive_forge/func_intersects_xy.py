import warnings
import numpy as np
from shapely import lib
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def intersects_xy(geom, x, y=None, **kwargs):
    """
    Returns True if A and the Point (x, y) share any portion of space.

    This is a special-case (and faster) variant of the `intersects` function
    which avoids having to create a Point object if you start from x/y
    coordinates.

    See the docstring of `intersects` for more details about the predicate.

    Parameters
    ----------
    geom : Geometry or array_like
    x, y : float or array_like
        Coordinates as separate x and y arrays, or a single array of
        coordinate x, y tuples.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    intersects : variant taking two geometries as input

    Notes
    -----
    If you compare a single or few geometries with many points, it can be
    beneficial to prepare the geometries in advance using
    :func:`shapely.prepare`.

    The `touches` predicate can be determined with this function by getting
    the boundary of the geometries: ``intersects_xy(boundary(geom), x, y)``.

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> line = LineString([(0, 0), (1, 1)])
    >>> intersects(line, Point(0, 0))
    True
    >>> intersects_xy(line, 0, 0)
    True
    """
    if y is None:
        coords = np.asarray(x)
        x, y = (coords[:, 0], coords[:, 1])
    return lib.intersects_xy(geom, x, y, **kwargs)