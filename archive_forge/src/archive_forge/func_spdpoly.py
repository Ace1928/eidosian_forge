import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def spdpoly(self, w, nma=50):
    """spectral density from MA polynomial representation for ARMA process

        References
        ----------
        Cochrane, section 8.3.3
        """
    mpoly = np.polynomial.Polynomial(self.arma2ma(nma))
    hw = mpoly(np.exp(1j * w))
    spd = np.real_if_close(hw * hw.conj() * 0.5 / np.pi)
    return (spd, w)