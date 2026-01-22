from shapely import lib
from shapely.decorators import multithreading_enabled
from shapely.errors import UnsupportedGEOSVersionError
@multithreading_enabled
def line_merge(line, directed=False, **kwargs):
    """Returns (Multi)LineStrings formed by combining the lines in a
    MultiLineString.

    Lines are joined together at their endpoints in case two lines are
    intersecting. Lines are not joined when 3 or more lines are intersecting at
    the endpoints. Line elements that cannot be joined are kept as is in the
    resulting MultiLineString.

    The direction of each merged LineString will be that of the majority of the
    LineStrings from which it was derived. Except if ``directed=True`` is
    specified, then the operation will not change the order of points within
    lines and so only lines which can be joined with no change in direction
    are merged.

    Parameters
    ----------
    line : Geometry or array_like
    directed : bool, default False
        Only combine lines if possible without changing point order.
        Requires GEOS >= 3.11.0
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import MultiLineString
    >>> line_merge(MultiLineString([[(0, 2), (0, 10)], [(0, 10), (5, 10)]]))
    <LINESTRING (0 2, 0 10, 5 10)>
    >>> line_merge(MultiLineString([[(0, 2), (0, 10)], [(0, 11), (5, 10)]]))
    <MULTILINESTRING ((0 2, 0 10), (0 11, 5 10))>
    >>> line_merge(MultiLineString())
    <GEOMETRYCOLLECTION EMPTY>
    >>> line_merge(MultiLineString([[(0, 0), (1, 0)], [(0, 0), (3, 0)]]))
    <LINESTRING (1 0, 0 0, 3 0)>
    >>> line_merge(MultiLineString([[(0, 0), (1, 0)], [(0, 0), (3, 0)]]), directed=True)
    <MULTILINESTRING ((0 0, 1 0), (0 0, 3 0))>
    """
    if directed:
        if lib.geos_version < (3, 11, 0):
            raise UnsupportedGEOSVersionError("'{}' requires at least GEOS {}.{}.{}.".format('line_merge', *(3, 11, 0)))
        return lib.line_merge_directed(line, **kwargs)
    return lib.line_merge(line, **kwargs)