from ._pnpoly import _grid_points_in_poly, _points_in_poly
def points_in_poly(points, verts):
    """Test whether points lie inside a polygon.

    Parameters
    ----------
    points : (K, 2) array
        Input points, ``(x, y)``.
    verts : (L, 2) array
        Vertices of the polygon, sorted either clockwise or anti-clockwise.
        The first point may (but does not need to be) duplicated.

    See Also
    --------
    grid_points_in_poly

    Returns
    -------
    mask : (K,) array of bool
        True if corresponding point is inside the polygon.

    """
    return _points_in_poly(points, verts)