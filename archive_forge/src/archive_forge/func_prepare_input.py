import cupy
from cupyx.scipy.interpolate._interpolate import PPoly
def prepare_input(x, y, axis, dydx=None):
    """Prepare input for cubic spline interpolators.
    All data are converted to numpy arrays and checked for correctness.
    Axes equal to `axis` of arrays `y` and `dydx` are moved to be the 0th
    axis. The value of `axis` is converted to lie in
    [0, number of dimensions of `y`).
    """
    x, y = map(cupy.asarray, (x, y))
    if cupy.issubdtype(x.dtype, cupy.complexfloating):
        raise ValueError('`x` must contain real values.')
    x = x.astype(float)
    if cupy.issubdtype(y.dtype, cupy.complexfloating):
        dtype = complex
    else:
        dtype = float
    if dydx is not None:
        dydx = cupy.asarray(dydx)
        if y.shape != dydx.shape:
            raise ValueError('The shapes of `y` and `dydx` must be identical.')
        if cupy.issubdtype(dydx.dtype, cupy.complexfloating):
            dtype = complex
        dydx = dydx.astype(dtype, copy=False)
    y = y.astype(dtype, copy=False)
    axis = axis % y.ndim
    if x.ndim != 1:
        raise ValueError('`x` must be 1-dimensional.')
    if x.shape[0] < 2:
        raise ValueError('`x` must contain at least 2 elements.')
    if x.shape[0] != y.shape[axis]:
        raise ValueError("The length of `y` along `axis`={0} doesn't match the length of `x`".format(axis))
    if not cupy.all(cupy.isfinite(x)):
        raise ValueError('`x` must contain only finite values.')
    if not cupy.all(cupy.isfinite(y)):
        raise ValueError('`y` must contain only finite values.')
    if dydx is not None and (not cupy.all(cupy.isfinite(dydx))):
        raise ValueError('`dydx` must contain only finite values.')
    dx = cupy.diff(x)
    if cupy.any(dx <= 0):
        raise ValueError('`x` must be strictly increasing sequence.')
    y = cupy.moveaxis(y, axis, 0)
    if dydx is not None:
        dydx = cupy.moveaxis(dydx, axis, 0)
    return (x, dx, y, axis, dydx)