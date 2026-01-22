from __future__ import annotations
from collections.abc import Iterator
from io import BytesIO
import warnings
import numpy as np
import numba as nb
import toolz as tz
import xarray as xr
import dask.array as da
from PIL.Image import fromarray
from datashader.colors import rgb, Sets1to3
from datashader.utils import nansum_missing, ngjit
def eq_hist(data, mask=None, nbins=256 * 256):
    """Compute the numpy array after histogram equalization.

    For use in `shade`.

    Parameters
    ----------
    data : ndarray
    mask : ndarray, optional
       Boolean array of missing points. Where True, the output will be `NaN`.
    nbins : int, optional
        Maximum number of bins to use. If data is of type boolean or integer
        this will determine when to switch from exact unique value counts to
        a binned histogram.

    Returns
    -------
    ndarray or tuple(ndarray, int)
        Returns the array when mask isn't set, otherwise returns the
        array and the computed number of discrete levels.

    Notes
    -----
    This function is adapted from the implementation in scikit-image [1]_.

    References
    ----------
    .. [1] http://scikit-image.org/docs/stable/api/skimage.exposure.html#equalize-hist
    """
    if cupy and isinstance(data, cupy.ndarray):
        from ._cuda_utils import interp
        array_module = cupy
    elif not isinstance(data, np.ndarray):
        raise TypeError('data must be an ndarray')
    else:
        interp = np.interp
        array_module = np
    if mask is not None and array_module.all(mask):
        return (array_module.full_like(data, np.nan), 0)
    data2 = data if mask is None else data[~mask]
    if data2.dtype == bool or (array_module.issubdtype(data2.dtype, array_module.integer) and data2.ptp() < nbins):
        values, counts = array_module.unique(data2, return_counts=True)
        vmin, vmax = (values[0].item(), values[-1].item())
        interval = vmax - vmin
        bin_centers = array_module.arange(vmin, vmax + 1)
        hist = array_module.zeros(interval + 1, dtype='uint64')
        hist[values - vmin] = counts
        discrete_levels = len(values)
    else:
        hist, bin_edges = array_module.histogram(data2, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        keep_mask = hist > 0
        discrete_levels = array_module.count_nonzero(keep_mask)
        if discrete_levels != len(hist):
            hist = hist[keep_mask]
            bin_centers = bin_centers[keep_mask]
    cdf = hist.cumsum()
    cdf = cdf / float(cdf[-1])
    out = interp(data, bin_centers, cdf).reshape(data.shape)
    return (out if mask is None else array_module.where(mask, array_module.nan, out), discrete_levels)