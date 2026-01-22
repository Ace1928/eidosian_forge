import numpy as np
import scipy.fftpack as fft
from scipy import signal
from statsmodels.tools.validation import array_like, PandasWrapper
def fftconvolve3(in1, in2=None, in3=None, mode='full'):
    """
    Convolve two N-dimensional arrays using FFT. See convolve.

    For use with arma  (old version: in1=num in2=den in3=data

    * better for consistency with other functions in1=data in2=num in3=den
    * note in2 and in3 need to have consistent dimension/shape
      since I'm using max of in2, in3 shapes and not the sum

    copied from scipy.signal.signaltools, but here used to try out inverse
    filter does not work or I cannot get it to work

    2010-10-23
    looks ok to me for 1d,
    from results below with padded data array (fftp)
    but it does not work for multidimensional inverse filter (fftn)
    original signal.fftconvolve also uses fftn
    """
    if in2 is None and in3 is None:
        raise ValueError('at least one of in2 and in3 needs to be given')
    s1 = np.array(in1.shape)
    if in2 is not None:
        s2 = np.array(in2.shape)
    else:
        s2 = 0
    if in3 is not None:
        s3 = np.array(in3.shape)
        s2 = max(s2, s3)
    complex_result = np.issubdtype(in1.dtype, np.complex) or np.issubdtype(in2.dtype, np.complex)
    size = s1 + s2 - 1
    fsize = 2 ** np.ceil(np.log2(size))
    IN1 = in1.copy()
    if in2 is not None:
        IN1 = fft.fftn(in2, fsize)
    if in3 is not None:
        IN1 /= fft.fftn(in3, fsize)
    IN1 *= fft.fftn(in1, fsize)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = fft.ifftn(IN1)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == 'full':
        return ret
    elif mode == 'same':
        if np.product(s1, axis=0) > np.product(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return trim_centered(ret, osize)
    elif mode == 'valid':
        return trim_centered(ret, abs(s2 - s1) + 1)