import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted
def fit_iterative(self, maxiter=3):
    """
        Perform an iterative two-step procedure to estimate the GLS model.

        Parameters
        ----------
        maxiter : int, optional
            the number of iterations

        Notes
        -----
        maxiter=1: returns the estimated based on given weights
        maxiter=2: performs a second estimation with the updated weights,
                   this is 2-step estimation
        maxiter>2: iteratively estimate and update the weights

        TODO: possible extension stop iteration if change in parameter
            estimates is smaller than x_tol

        Repeated calls to fit_iterative, will do one redundant pinv_wexog
        calculation. Calling fit_iterative(maxiter) once does not do any
        redundant recalculations (whitening or calculating pinv_wexog).
        """
    if maxiter < 1:
        raise ValueError('maxiter needs to be at least 1')
    import collections
    self.history = collections.defaultdict(list)
    for i in range(maxiter):
        if hasattr(self, 'pinv_wexog'):
            del self.pinv_wexog
        results = self.fit()
        self.history['self_params'].append(results.params)
        if not i == maxiter - 1:
            self.results_old = results
            sigma_i = self.get_within_cov(results.resid)
            self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
            self.initialize()
    return results