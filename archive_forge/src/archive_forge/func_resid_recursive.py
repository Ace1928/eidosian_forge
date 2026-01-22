import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import (
from statsmodels.tsa.statespace.tools import concat
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
@cache_readonly
def resid_recursive(self):
    """
        Recursive residuals

        Returns
        -------
        resid_recursive : array_like
            An array of length `nobs` holding the recursive
            residuals.

        Notes
        -----
        These quantities are defined in, for example, Harvey (1989)
        section 5.4. In fact, there he defines the standardized innovations in
        equation 5.4.1, but in his version they have non-unit variance, whereas
        the standardized forecast errors computed by the Kalman filter here
        assume unit variance. To convert to Harvey's definition, we need to
        multiply by the standard deviation.

        Harvey notes that in smaller samples, "although the second moment
        of the :math:`\\tilde \\sigma_*^{-1} \\tilde v_t`'s is unity, the
        variance is not necessarily equal to unity as the mean need not be
        equal to zero", and he defines an alternative version (which are
        not provided here).
        """
    return self.filter_results.standardized_forecasts_error[0] * self.scale ** 0.5