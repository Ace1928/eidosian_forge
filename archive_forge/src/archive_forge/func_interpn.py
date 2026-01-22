import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator
def interpn(points, values, xi, method='linear', bounds_error=True, fill_value=cp.nan):
    """
    Multidimensional interpolation on regular or rectilinear grids.

    Strictly speaking, not all regular grids are supported - this function
    works on *rectilinear* grids, that is, a rectangular grid with even or
    uneven spacing.

    Parameters
    ----------
    points : tuple of cupy.ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions. The points in
        each dimension (i.e. every elements of the points tuple) must be
        strictly ascending or descending.

    values : cupy.ndarray of shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions. Complex data can be
        acceptable.

    xi : cupy.ndarray of shape (..., ndim)
        The coordinates to sample the gridded data at

    method : str, optional
        The method of interpolation to perform. Supported are "linear",
        "nearest", "slinear", "cubic", "quintic" and "pchip".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.

    Returns
    -------
    values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
        Interpolated values at `xi`. See notes for behaviour when
        ``xi.ndim == 1``.

    Notes
    -----

    In the case that ``xi.ndim == 1`` a new axis is inserted into
    the 0 position of the returned array, values_x, so its shape is
    instead ``(1,) + values.shape[ndim:]``.

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolation.

    Examples
    --------
    Evaluate a simple example function on the points of a regular 3-D grid:

    >>> import cupy as cp
    >>> from cupyx.scipy.interpolate import interpn
    >>> def value_func_3d(x, y, z):
    ...     return 2 * x + 3 * y - z
    >>> x = cp.linspace(0, 4, 5)
    >>> y = cp.linspace(0, 5, 6)
    >>> z = cp.linspace(0, 6, 7)
    >>> points = (x, y, z)
    >>> values = value_func_3d(*cp.meshgrid(*points, indexing='ij'))

    Evaluate the interpolating function at a point

    >>> point = cp.array([2.21, 3.12, 1.15])
    >>> print(interpn(points, values, point))
    [12.63]

    See Also
    --------
    RegularGridInterpolator : interpolation on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    cupyx.scipy.ndimage.map_coordinates : interpolation on grids with equal
                                          spacing (suitable for e.g., N-D image
                                          resampling)
    """
    if method not in ['linear', 'nearest', 'slinear', 'cubic', 'quintic', 'pchip']:
        raise ValueError("interpn only understands the methods 'linear', 'nearest', 'slinear', 'cubic', 'quintic' and 'pchip'. You provided {method}.")
    ndim = values.ndim
    if len(points) > ndim:
        raise ValueError('There are %d point arrays, but values has %d dimensions' % (len(points), ndim))
    grid, descending_dimensions = _check_points(points)
    _check_dimensionality(grid, values)
    xi = _ndim_coords_from_arrays(xi, ndim=len(grid))
    if xi.shape[-1] != len(grid):
        raise ValueError('The requested sample points xi have dimension %d, but this RegularGridInterpolator has dimension %d' % (xi.shape[-1], len(grid)))
    if bounds_error:
        for i, p in enumerate(xi.T):
            if not cp.logical_and(cp.all(grid[i][0] <= p), cp.all(p <= grid[i][-1])):
                raise ValueError('One of the requested xi is out of bounds in dimension %d' % i)
    if method in ['linear', 'nearest', 'slinear', 'cubic', 'quintic', 'pchip']:
        interp = RegularGridInterpolator(points, values, method=method, bounds_error=bounds_error, fill_value=fill_value)
        return interp(xi)