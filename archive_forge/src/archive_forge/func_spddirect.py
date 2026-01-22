import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def spddirect(self, n):
    """power spectral density using padding to length n done by fft

        currently returns two-sided according to fft frequencies, use first half
        """
    hw = fft.fft(self.ma, n) / fft.fft(self.ar, n)
    w = fft.fftfreq(n) * 2 * np.pi
    wslice = slice(None, n // 2, None)
    return (np.abs(hw) ** 2 * 0.5 / np.pi, w)