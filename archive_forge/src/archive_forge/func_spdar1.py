import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def spdar1(ar, w):
    if np.ndim(ar) == 0:
        rho = ar
    else:
        rho = -ar[1]
    return 0.5 / np.pi / (1 + rho * rho - 2 * rho * np.cos(w))