import numpy as np
from scipy.interpolate import interp1d, interp2d, Rbf
from statsmodels.tools.decorators import cache_readonly
def crit(self, prob, n):
    """
        Returns interpolated quantiles, similar to ppf or isf

        use two sequential 1d interpolation, first by n then by prob

        Parameters
        ----------
        prob : array_like
            probabilities corresponding to the definition of table columns
        n : int or float
            sample size, second parameter of the table

        Returns
        -------
        ppf : array_like
            critical values with same shape as prob
        """
    prob = np.asarray(prob)
    alpha = self.alpha
    critv = self._critvals(n)
    cond_ilow = prob > alpha[0]
    cond_ihigh = prob < alpha[-1]
    cond_interior = np.logical_or(cond_ilow, cond_ihigh)
    if prob.size == 1:
        if cond_interior:
            return interp1d(alpha, critv)(prob)
        else:
            return np.nan
    quantile = np.nan * np.ones(prob.shape)
    quantile[cond_interior] = interp1d(alpha, critv)(prob[cond_interior])
    return quantile