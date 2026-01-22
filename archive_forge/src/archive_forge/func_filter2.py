import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def filter2(self, x, pad=0):
    """filter a time series using fftconvolve3 with ARMA filter

        padding of x currently works only if x is 1d
        in example it produces same observations at beginning as lfilter even
        without padding.

        TODO: this returns 1 additional observation at the end
        """
    from statsmodels.tsa.filters import fftconvolve3
    if not pad:
        pass
    elif pad == 'auto':
        x = self.padarr(x, x.shape[0] + 2 * (self.nma + self.nar), atend=False)
    else:
        x = self.padarr(x, x.shape[0] + int(pad), atend=False)
    return fftconvolve3(x, self.ma, self.ar)