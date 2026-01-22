import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def to_ragged_array(geometries, include_z=None):
    """
    Converts geometries to a ragged array representation using a contiguous
    array of coordinates and offset arrays.

    This function converts an array of geometries to a ragged array
    (i.e. irregular array of arrays) of coordinates, represented in memory
    using a single contiguous array of the coordinates, and
    up to 3 offset arrays that keep track where each sub-array
    starts and ends.

    This follows the in-memory layout of the variable size list arrays defined
    by Apache Arrow, as specified for geometries by the GeoArrow project:
    https://github.com/geoarrow/geoarrow.

    Parameters
    ----------
    geometries : array_like
        Array of geometries (1-dimensional).
    include_z : bool, default None
        If False, return 2D geometries. If True, include the third dimension
        in the output (if a geometry has no third dimension, the z-coordinates
        will be NaN). By default, will infer the dimensionality from the
        input geometries. Note that this inference can be unreliable with
        empty geometries (for a guaranteed result, it is recommended to
        specify the keyword).

    Returns
    -------
    tuple of (geometry_type, coords, offsets)
        geometry_type : GeometryType
            The type of the input geometries (required information for
            roundtrip).
        coords : np.ndarray
            Contiguous array of shape (n, 2) or (n, 3) of all coordinates
            of all input geometries.
        offsets: tuple of np.ndarray
            Offset arrays that make it possible to reconstruct the
            geometries from the flat coordinates array. The number of
            offset arrays depends on the geometry type. See
            https://github.com/geoarrow/geoarrow/blob/main/format.md
            for details.

    Notes
    -----
    Mixed singular and multi geometry types of the same basic type are
    allowed (e.g., Point and MultiPoint) and all singular types will be
    treated as multi types.
    GeometryCollections and other mixed geometry types are not supported.

    See also
    --------
    from_ragged_array

    Examples
    --------
    Consider a Polygon with one hole (interior ring):

    >>> import shapely
    >>> polygon = shapely.Polygon(
    ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
    ...     holes=[[(2, 2), (3, 2), (2, 3)]]
    ... )
    >>> polygon
    <POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 3 2, 2 3, 2 2))>

    This polygon can be thought of as a list of rings (first ring is the
    exterior ring, subsequent rings are the interior rings), and each ring
    as a list of coordinate pairs. This is very similar to how GeoJSON
    represents the coordinates:

    >>> import json
    >>> json.loads(shapely.to_geojson(polygon))["coordinates"]
    [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
     [[2.0, 2.0], [3.0, 2.0], [2.0, 3.0], [2.0, 2.0]]]

    This function will return a similar list of lists of lists, but
    using a single contiguous array of coordinates, and multiple arrays of
    offsets:

    >>> geometry_type, coords, offsets = shapely.to_ragged_array([polygon])
    >>> geometry_type
    <GeometryType.POLYGON: 3>
    >>> coords
    array([[ 0.,  0.],
           [10.,  0.],
           [10., 10.],
           [ 0., 10.],
           [ 0.,  0.],
           [ 2.,  2.],
           [ 3.,  2.],
           [ 2.,  3.],
           [ 2.,  2.]])

    >>> offsets
    (array([0, 5, 9]), array([0, 2]))

    As an example how to interpret the offsets: the i-th ring in the
    coordinates is represented by ``offsets[0][i]`` to ``offsets[0][i+1]``:

    >>> exterior_ring_start, exterior_ring_end = offsets[0][0], offsets[0][1]
    >>> coords[exterior_ring_start:exterior_ring_end]
    array([[ 0.,  0.],
           [10.,  0.],
           [10., 10.],
           [ 0., 10.],
           [ 0.,  0.]])

    """
    geometries = np.asarray(geometries)
    if include_z is None:
        include_z = np.any(get_coordinate_dimension(geometries[~is_empty(geometries)]) == 3)
    geom_types = np.unique(get_type_id(geometries))
    geom_types = geom_types[geom_types >= 0]
    if len(geom_types) == 1:
        typ = GeometryType(geom_types[0])
        if typ == GeometryType.POINT:
            coords, offsets = _get_arrays_point(geometries, include_z)
        elif typ == GeometryType.LINESTRING:
            coords, offsets = _get_arrays_linestring(geometries, include_z)
        elif typ == GeometryType.POLYGON:
            coords, offsets = _get_arrays_polygon(geometries, include_z)
        elif typ == GeometryType.MULTIPOINT:
            coords, offsets = _get_arrays_multipoint(geometries, include_z)
        elif typ == GeometryType.MULTILINESTRING:
            coords, offsets = _get_arrays_multilinestring(geometries, include_z)
        elif typ == GeometryType.MULTIPOLYGON:
            coords, offsets = _get_arrays_multipolygon(geometries, include_z)
        else:
            raise ValueError(f'Geometry type {typ.name} is not supported')
    elif len(geom_types) == 2:
        if set(geom_types) == {GeometryType.POINT, GeometryType.MULTIPOINT}:
            typ = GeometryType.MULTIPOINT
            coords, offsets = _get_arrays_multipoint(geometries, include_z)
        elif set(geom_types) == {GeometryType.LINESTRING, GeometryType.MULTILINESTRING}:
            typ = GeometryType.MULTILINESTRING
            coords, offsets = _get_arrays_multilinestring(geometries, include_z)
        elif set(geom_types) == {GeometryType.POLYGON, GeometryType.MULTIPOLYGON}:
            typ = GeometryType.MULTIPOLYGON
            coords, offsets = _get_arrays_multipolygon(geometries, include_z)
        else:
            raise ValueError(f'Geometry type combination is not supported ({[GeometryType(t).name for t in geom_types]})')
    else:
        raise ValueError(f'Geometry type combination is not supported ({[GeometryType(t).name for t in geom_types]})')
    return (typ, coords, offsets)