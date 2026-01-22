import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.tsa.arima_process import arma_generate_sample, arma_impulse_response
from statsmodels.tsa.arima_process import arma_acovf, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf
def norm_corr(x, y, mode='valid'):
    """Returns the correlation between two ndarrays, by calling np.correlate in
'same' mode and normalizing the result by the std of the arrays and by
their lengths. This results in a correlation = 1 for an auto-correlation"""
    return np.correlate(x, y, mode) / (np.std(x) * np.std(y) * x.shape[-1])