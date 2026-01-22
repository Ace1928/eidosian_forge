from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
@property
def isstationary(self):
    """
        Arma process is stationary if AR roots are outside unit circle.

        Returns
        -------
        bool
             True if autoregressive roots are outside unit circle.
        """
    if np.all(np.abs(self.arroots) > 1.0):
        return True
    else:
        return False