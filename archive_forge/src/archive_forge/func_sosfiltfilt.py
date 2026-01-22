import operator
import math
from math import prod as _prod
import timeit
import warnings
from scipy.spatial import cKDTree
from . import _sigtools
from ._ltisys import dlti
from ._upfirdn import upfirdn, _output_len, _upfirdn_modes
from scipy import linalg, fft as sp_fft
from scipy import ndimage
from scipy.fft._helper import _init_nd_shape_and_axes
import numpy as np
from scipy.special import lambertw
from .windows import get_window
from ._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext
from ._filter_design import cheby1, _validate_sos, zpk2sos
from ._fir_filter_design import firwin
from ._sosfilt import _sosfilt
def sosfiltfilt(sos, x, axis=-1, padtype='odd', padlen=None):
    """
    A forward-backward digital filter using cascaded second-order sections.

    See `filtfilt` for more complete information about this method.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is::

            3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                        (sos[:, 5] == 0).sum()))

        The extra subtraction at the end attempts to compensate for poles
        and zeros at the origin (e.g. for odd-order filters) to yield
        equivalent estimates of `padlen` to those of `filtfilt` for
        second-order section filters built with `scipy.signal` functions.

    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.

    See Also
    --------
    filtfilt, sosfilt, sosfilt_zi, sosfreqz

    Notes
    -----
    .. versionadded:: 0.18.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import sosfiltfilt, butter
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    Create an interesting signal to filter.

    >>> n = 201
    >>> t = np.linspace(0, 1, n)
    >>> x = 1 + (t < 0.5) - 0.25*t**2 + 0.05*rng.standard_normal(n)

    Create a lowpass Butterworth filter, and use it to filter `x`.

    >>> sos = butter(4, 0.125, output='sos')
    >>> y = sosfiltfilt(sos, x)

    For comparison, apply an 8th order filter using `sosfilt`.  The filter
    is initialized using the mean of the first four values of `x`.

    >>> from scipy.signal import sosfilt, sosfilt_zi
    >>> sos8 = butter(8, 0.125, output='sos')
    >>> zi = x[:4].mean() * sosfilt_zi(sos8)
    >>> y2, zo = sosfilt(sos8, x, zi=zi)

    Plot the results.  Note that the phase of `y` matches the input, while
    `y2` has a significant phase delay.

    >>> plt.plot(t, x, alpha=0.5, label='x(t)')
    >>> plt.plot(t, y, label='y(t)')
    >>> plt.plot(t, y2, label='y2(t)')
    >>> plt.legend(framealpha=1, shadow=True)
    >>> plt.grid(alpha=0.25)
    >>> plt.xlabel('t')
    >>> plt.show()

    """
    sos, n_sections = _validate_sos(sos)
    x = _validate_x(x)
    ntaps = 2 * n_sections + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    edge, ext = _validate_pad(padtype, padlen, x, axis, ntaps=ntaps)
    zi = sosfilt_zi(sos)
    zi_shape = [1] * x.ndim
    zi_shape[axis] = 2
    zi.shape = [n_sections] + zi_shape
    x_0 = axis_slice(ext, stop=1, axis=axis)
    y, zf = sosfilt(sos, ext, axis=axis, zi=zi * x_0)
    y_0 = axis_slice(y, start=-1, axis=axis)
    y, zf = sosfilt(sos, axis_reverse(y, axis=axis), axis=axis, zi=zi * y_0)
    y = axis_reverse(y, axis=axis)
    if edge > 0:
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)
    return y