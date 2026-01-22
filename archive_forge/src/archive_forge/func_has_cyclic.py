import numpy as np
import numpy.ma as ma
def has_cyclic(x, axis=-1, cyclic=360, precision=0.0001):
    """
    Check if x/longitude coordinates already have a cyclic point.

    Checks all differences between the first and last
    x-coordinates along ``axis`` to be less than ``precision``.

    Parameters
    ----------
    x : ndarray
        An array with the x-coordinate values to be checked for cyclic points.
    axis : int, optional
        Specifies the axis of the ``x`` array to be checked.
        Defaults to the right-most axis.
    cyclic : float, optional
        Width of periodic domain (default: 360).
    precision : float, optional
        Maximal difference between first and last x-coordinate to detect
        cyclic point (default: 1e-4).

    Returns
    -------
    True if a cyclic point was detected along the given axis,
    False otherwise.

    Examples
    --------
    Check for cyclic x-coordinate in one dimension.
    >>> import numpy as np
    >>> lons = np.arange(0, 360, 60)
    >>> clons = np.arange(0, 361, 60)
    >>> print(has_cyclic(lons))
    False
    >>> print(has_cyclic(clons))
    True

    Check for cyclic x-coordinate in two dimensions.
    >>> lats = np.arange(-90, 90, 30)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> clon2d, clat2d = np.meshgrid(clons, lats)
    >>> print(has_cyclic(lon2d))
    False
    >>> print(has_cyclic(clon2d))
    True

    """
    npc = np.ma if np.ma.is_masked(x) else np
    x1 = np.mod(npc.where(x < 0, x + cyclic, x), cyclic)
    dd = np.diff(np.take(x1, [0, -1], axis=axis), axis=axis)
    if npc.all(np.abs(dd) < precision):
        return True
    else:
        return False