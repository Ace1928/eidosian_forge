import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def spd(self, npos):
    """raw spectral density, returns Fourier transform

        n is number of points in positive spectrum, the actual number of points
        is twice as large. different from other spd methods with fft
        """
    n = npos
    w = fft.fftfreq(2 * n) * 2 * np.pi
    hw = self.fftarma(2 * n)
    return ((hw * hw.conj()).real * 0.5 / np.pi, w)