from CuSignal under terms of the MIT license.
import warnings
from typing import Set
import cupy
import numpy as np
def get_window(window, Nx, fftbins=True):
    """
    Return a window of a given length and type.

    Parameters
    ----------
    window : string, float, or tuple
        The type of window to create. See below for more details.
    Nx : int
        The number of samples in the window.
    fftbins : bool, optional
        If True (default), create a "periodic" window, ready to use with
        `ifftshift` and be multiplied by the result of an FFT (see also
        `fftpack.fftfreq`).
        If False, create a "symmetric" window, for use in filter design.

    Returns
    -------
    get_window : ndarray
        Returns a window of length `Nx` and type `window`

    Notes
    -----
    Window types:

    - :func:`~cupyx.scipy.signal.windows.boxcar`
    - :func:`~cupyx.scipy.signal.windows.triang`
    - :func:`~cupyx.scipy.signal.windows.blackman`
    - :func:`~cupyx.scipy.signal.windows.hamming`
    - :func:`~cupyx.scipy.signal.windows.hann`
    - :func:`~cupyx.scipy.signal.windows.bartlett`
    - :func:`~cupyx.scipy.signal.windows.flattop`
    - :func:`~cupyx.scipy.signal.windows.parzen`
    - :func:`~cupyx.scipy.signal.windows.bohman`
    - :func:`~cupyx.scipy.signal.windows.blackmanharris`
    - :func:`~cupyx.scipy.signal.windows.nuttall`
    - :func:`~cupyx.scipy.signal.windows.barthann`
    - :func:`~cupyx.scipy.signal.windows.kaiser` (needs beta)
    - :func:`~cupyx.scipy.signal.windows.gaussian` (needs standard deviation)
    - :func:`~cupyx.scipy.signal.windows.general_gaussian` (needs power, width)
    - :func:`~cupyx.scipy.signal.windows.chebwin` (needs attenuation)
    - :func:`~cupyx.scipy.signal.windows.exponential` (needs decay scale)
    - :func:`~cupyx.scipy.signal.windows.tukey` (needs taper fraction)

    If the window requires no parameters, then `window` can be a string.

    If the window requires parameters, then `window` must be a tuple
    with the first argument the string name of the window, and the next
    arguments the needed parameters.

    If `window` is a floating point number, it is interpreted as the beta
    parameter of the :func:`~cupyx.scipy.signal.windows.kaiser` window.

    Each of the window types listed above is also the name of
    a function that can be called directly to create a window of
    that type.

    Examples
    --------
    >>> import cupyx.scipy.signal.windows
    >>> cupyx.scipy.signal.windows.get_window('triang', 7)
    array([ 0.125,  0.375,  0.625,  0.875,  0.875,  0.625,  0.375])
    >>> cupyx.scipy.signal.windows.get_window(('kaiser', 4.0), 9)
    array([0.08848053, 0.32578323, 0.63343178, 0.89640418, 1.,
           0.89640418, 0.63343178, 0.32578323, 0.08848053])
    >>> cupyx.scipy.signal.windows.get_window(4.0, 9)
    array([0.08848053, 0.32578323, 0.63343178, 0.89640418, 1.,
           0.89640418, 0.63343178, 0.32578323, 0.08848053])

    """
    sym = not fftbins
    try:
        beta = float(window)
    except (TypeError, ValueError):
        args = ()
        if isinstance(window, tuple):
            winstr = window[0]
            if len(window) > 1:
                args = window[1:]
        elif isinstance(window, str):
            if window in _needs_param:
                raise ValueError("The '" + window + "' window needs one or more parameters -- pass a tuple.")
            else:
                winstr = window
        else:
            raise ValueError('%s as window type is not supported.' % str(type(window)))
        try:
            winfunc = _win_equiv[winstr]
        except KeyError:
            raise ValueError('Unknown window type.')
        params = (Nx,) + args + (sym,)
    else:
        winfunc = kaiser
        params = (Nx, beta, sym)
    return winfunc(*params)