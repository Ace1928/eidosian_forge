import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def spdmapoly(self, w, twosided=False):
    """ma only, need division for ar, use LagPolynomial
        """
    if w is None:
        w = np.linspace(0, np.pi, nfreq)
    return 0.5 / np.pi * self.mapoly(np.exp(w * 1j))