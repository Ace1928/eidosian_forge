import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, RegressionResults
from statsmodels.tools.tools import add_constant
def conf_int_el(self, param_num, upper_bound=None, lower_bound=None, sig=0.05, method='nm', stochastic_exog=True):
    """
        Returns the confidence interval for a regression parameter when the
        regression is forced through the origin.

        Parameters
        ----------
        param_num : int
            The parameter number to be tested.  Note this uses python
            indexing but the '0' parameter refers to the intercept term.
        upper_bound : float
            The maximum value the upper confidence limit can be.  The
            closer this is to the confidence limit, the quicker the
            computation.  Default is .00001 confidence limit under normality.
        lower_bound : float
            The minimum value the lower confidence limit can be.
            Default is .00001 confidence limit under normality.
        sig : float, optional
            The significance level.  Default .05.
        method : str, optional
             Algorithm to optimize of nuisance params.  Can be 'nm' or
            'powell'.  Default is 'nm'.
        stochastic_exog : bool
            Default is True.

        Returns
        -------
        ci: tuple
            The confidence interval for the parameter 'param_num'.
        """
    r0 = chi2.ppf(1 - sig, 1)
    param_num = np.array([param_num])
    if upper_bound is None:
        ci = np.asarray(self.model.fit().conf_int(0.0001))
        upper_bound = np.squeeze(ci[param_num])[1]
    if lower_bound is None:
        ci = np.asarray(self.model.fit().conf_int(0.0001))
        lower_bound = np.squeeze(ci[param_num])[0]

    def f(b0):
        b0 = np.array([b0])
        val = self.el_test(b0, param_num, method=method, stochastic_exog=stochastic_exog)
        return val[0] - r0
    _param = np.squeeze(self.params[param_num])
    lowerl = optimize.brentq(f, np.squeeze(lower_bound), _param)
    upperl = optimize.brentq(f, _param, np.squeeze(upper_bound))
    return (lowerl, upperl)