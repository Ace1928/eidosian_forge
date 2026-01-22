from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS, GLS, RegressionResults
from statsmodels.regression.feasible_gls import atleast_2dcols
def select_pen_weight(self, method='aicc', start_params=1.0, optim_args=None):
    """find penalization factor that minimizes gcv or an information criterion

        Parameters
        ----------
        method : str
            the name of an attribute of the results class. Currently the following
            are available aic, aicc, bic, gc and gcv.
        start_params : float
            starting values for the minimization to find the penalization factor
            `lambd`. Not since there can be local minima, it is best to try
            different starting values.
        optim_args : None or dict
            optimization keyword arguments used with `scipy.optimize.fmin`

        Returns
        -------
        min_pen_weight : float
            The penalization factor at which the target criterion is (locally)
            minimized.

        Notes
        -----
        This uses `scipy.optimize.fmin` as optimizer.
        """
    if optim_args is None:
        optim_args = {}

    def get_ic(lambd):
        return getattr(self.fit(lambd), method)
    from scipy import optimize
    lambd = optimize.fmin(get_ic, start_params, **optim_args)
    return lambd