from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
def lpol2index(ar):
    """
    Remove zeros from lag polynomial

    Parameters
    ----------
    ar : array_like
        coefficients of lag polynomial

    Returns
    -------
    coeffs : ndarray
        non-zero coefficients of lag polynomial
    index : ndarray
        index (lags) of lag polynomial with non-zero elements
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ComplexWarning)
        ar = array_like(ar, 'ar')
    index = np.nonzero(ar)[0]
    coeffs = ar[index]
    return (coeffs, index)