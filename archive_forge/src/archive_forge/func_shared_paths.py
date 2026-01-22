from shapely import lib
from shapely.decorators import multithreading_enabled
from shapely.errors import UnsupportedGEOSVersionError
@multithreading_enabled
def shared_paths(a, b, **kwargs):
    """Returns the shared paths between geom1 and geom2.

    Both geometries should be linestrings or arrays of linestrings.
    A geometrycollection or array of geometrycollections is returned
    with two elements in each geometrycollection. The first element is a
    multilinestring containing shared paths with the same direction
    for both inputs. The second element is a multilinestring containing
    shared paths with the opposite direction for the two inputs.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import LineString
    >>> line1 = LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> line2 = LineString([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)])
    >>> shared_paths(line1, line2).wkt
    'GEOMETRYCOLLECTION (MULTILINESTRING EMPTY, MULTILINESTRING ((1 0, 1 1)))'
    >>> line3 = LineString([(1, 1), (0, 1)])
    >>> shared_paths(line1, line3).wkt
    'GEOMETRYCOLLECTION (MULTILINESTRING ((1 1, 0 1)), MULTILINESTRING EMPTY)'
    """
    return lib.shared_paths(a, b, **kwargs)