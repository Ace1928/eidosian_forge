import operator
from math import gcd
import cupy
from cupyx.scipy.fft import fft, rfft, fftfreq, ifft, irfft, ifftshift
from cupyx.scipy.signal._iir_filter_design import cheby1
from cupyx.scipy.signal._fir_filter_design import firwin
from cupyx.scipy.signal._iir_filter_conversions import zpk2sos
from cupyx.scipy.signal._ltisys import dlti
from cupyx.scipy.signal._upfirdn import upfirdn, _output_len
from cupyx.scipy.signal._signaltools import (
from cupyx.scipy.signal.windows._windows import get_window
def decimate(x, q, n=None, ftype='iir', axis=-1, zero_phase=True):
    """
    Downsample the signal after applying an anti-aliasing filter.

    By default, an order 8 Chebyshev type I filter is used. A 30 point FIR
    filter with Hamming window is used if `ftype` is 'fir'.

    Parameters
    ----------
    x : array_like
        The signal to be downsampled, as an N-dimensional array.
    q : int
        The downsampling factor. When using IIR downsampling, it is recommended
        to call `decimate` multiple times for downsampling factors higher than
        13.
    n : int, optional
        The order of the filter (1 less than the length for 'fir'). Defaults to
        8 for 'iir' and 20 times the downsampling factor for 'fir'.
    ftype : str {'iir', 'fir'} or ``dlti`` instance, optional
        If 'iir' or 'fir', specifies the type of lowpass filter. If an instance
        of an `dlti` object, uses that object to filter before downsampling.
    axis : int, optional
        The axis along which to decimate.
    zero_phase : bool, optional
        Prevent phase shift by filtering with `filtfilt` instead of `lfilter`
        when using an IIR filter, and shifting the outputs back by the filter's
        group delay when using an FIR filter. The default value of ``True`` is
        recommended, since a phase shift is generally not desired.

    Returns
    -------
    y : ndarray
        The down-sampled signal.

    See Also
    --------
    resample : Resample up or down using the FFT method.
    resample_poly : Resample using polyphase filtering and an FIR filter.
    """
    x = cupy.asarray(x)
    q = operator.index(q)
    if n is not None:
        n = operator.index(n)
    result_type = x.dtype
    if not cupy.issubdtype(result_type, cupy.inexact) or result_type.type == cupy.float16:
        result_type = cupy.float64
    if ftype == 'fir':
        if n is None:
            half_len = 10 * q
            n = 2 * half_len
        b, a = (firwin(n + 1, 1.0 / q, window='hamming'), 1.0)
        b = cupy.asarray(b, dtype=result_type)
        a = cupy.asarray(a, dtype=result_type)
    elif ftype == 'iir':
        iir_use_sos = True
        if n is None:
            n = 8
        sos = cheby1(n, 0.05, 0.8 / q, output='sos')
        sos = cupy.asarray(sos, dtype=result_type)
    elif isinstance(ftype, dlti):
        system = ftype._as_zpk()
        if system.poles.shape[0] == 0:
            system = ftype._as_tf()
            b, a = (system.num, system.den)
            ftype = 'fir'
        elif any(cupy.iscomplex(system.poles)) or any(cupy.iscomplex(system.poles)) or cupy.iscomplex(system.gain):
            iir_use_sos = False
            system = ftype._as_tf()
            b, a = (system.num, system.den)
        else:
            iir_use_sos = True
            sos = zpk2sos(system.zeros, system.poles, system.gain)
            sos = cupy.asarray(sos, dtype=result_type)
    else:
        raise ValueError('invalid ftype')
    sl = [slice(None)] * x.ndim
    if ftype == 'fir':
        b = b / a
        if zero_phase:
            y = resample_poly(x, 1, q, axis=axis, window=b)
        else:
            n_out = x.shape[axis] // q + bool(x.shape[axis] % q)
            y = upfirdn(b, x, up=1, down=q, axis=axis)
            sl[axis] = slice(None, n_out, None)
    else:
        if zero_phase:
            if iir_use_sos:
                y = sosfiltfilt(sos, x, axis=axis)
            else:
                y = filtfilt(b, a, x, axis=axis)
        elif iir_use_sos:
            y = sosfilt(sos, x, axis=axis)
        else:
            y = lfilter(b, a, x, axis=axis)
        sl[axis] = slice(None, None, q)
    return y[tuple(sl)]