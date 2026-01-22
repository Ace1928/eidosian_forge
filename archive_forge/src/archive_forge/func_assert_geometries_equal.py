from functools import partial
import numpy as np
import shapely
def assert_geometries_equal(x, y, tolerance=1e-07, equal_none=True, equal_nan=True, normalize=False, err_msg='', verbose=True):
    """Raises an AssertionError if two geometry array_like objects are not equal.

    Given two array_like objects, check that the shape is equal and all elements of
    these objects are equal. An exception is raised at shape mismatch or conflicting
    values. In contrast to the standard usage in shapely, no assertion is raised if
    both objects have NaNs/Nones in the same positions.

    Parameters
    ----------
    x : Geometry or array_like
    y : Geometry or array_like
    equal_none : bool, default True
        Whether to consider None elements equal to other None elements.
    equal_nan : bool, default True
        Whether to consider nan coordinates as equal to other nan coordinates.
    normalize : bool, default False
        Whether to normalize geometries prior to comparison.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    """
    __tracebackhide__ = True
    if normalize:
        x = shapely.normalize(x)
        y = shapely.normalize(y)
    x = np.asarray(x)
    y = np.asarray(y)
    is_scalar = x.ndim == 0 or y.ndim == 0
    if not (is_scalar or x.shape == y.shape):
        msg = build_err_msg([x, y], err_msg + f'\n(shapes {x.shape}, {y.shape} mismatch)', verbose=verbose)
        raise AssertionError(msg)
    flagged = False
    if equal_none:
        flagged = _assert_none_same(x, y, err_msg, verbose)
    if not np.isscalar(flagged):
        x, y = (x[~flagged], y[~flagged])
        if x.size == 0:
            return
    elif flagged:
        return
    is_equal = _equals_exact_with_ndim(x, y, tolerance=tolerance)
    if is_scalar and (not np.isscalar(is_equal)):
        is_equal = bool(is_equal[0])
    if np.all(is_equal):
        return
    elif not equal_nan:
        msg = build_err_msg([x, y], err_msg + f'\nNot equal to tolerance {tolerance:g}', verbose=verbose)
        raise AssertionError(msg)
    if not np.isscalar(is_equal):
        x, y = (x[~is_equal], y[~is_equal])
        if x.size == 0:
            return
    elif is_equal:
        return
    is_equal = _assert_nan_coords_same(x, y, tolerance, err_msg, verbose)
    if not np.all(is_equal):
        msg = build_err_msg([x, y], err_msg + f'\nNot equal to tolerance {tolerance:g}', verbose=verbose)
        raise AssertionError(msg)