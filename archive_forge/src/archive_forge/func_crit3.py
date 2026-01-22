import numpy as np
from scipy.interpolate import interp1d, interp2d, Rbf
from statsmodels.tools.decorators import cache_readonly
def crit3(self, prob, n):
    """
        Returns interpolated quantiles, similar to ppf or isf

        uses Rbf to interpolate critical values as function of `prob` and `n`

        Parameters
        ----------
        prob : array_like
            probabilities corresponding to the definition of table columns
        n : int or float
            sample size, second parameter of the table

        Returns
        -------
        ppf : array_like
            critical values with same shape as prob, returns nan for arguments
            that are outside of the table bounds
        """
    prob = np.asarray(prob)
    alpha = self.alpha
    cond_ilow = prob > alpha[0]
    cond_ihigh = prob < alpha[-1]
    cond_interior = np.logical_or(cond_ilow, cond_ihigh)
    if prob.size == 1:
        if cond_interior:
            return self.polyrbf(n, prob)
        else:
            return np.nan
    quantile = np.nan * np.ones(prob.shape)
    quantile[cond_interior] = self.polyrbf(n, prob[cond_interior])
    return quantile