import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from statsmodels.nonparametric.api import KDEMultivariate, KernelReg
from statsmodels.nonparametric._kernel_base import \
def cv_loo(self, params):
    """
        Similar to the cross validation leave-one-out estimator.

        Modified to reflect the linear components.

        Parameters
        ----------
        params : array_like
            Vector consisting of the coefficients (b) and the bandwidths (bw).
            The first ``k_linear`` elements are the coefficients.

        Returns
        -------
        L : float
            The value of the objective function

        References
        ----------
        See p.254 in [1]
        """
    params = np.asarray(params)
    b = params[0:self.k_linear]
    bw = params[self.k_linear:]
    LOO_X = LeaveOneOut(self.exog)
    LOO_Y = LeaveOneOut(self.endog).__iter__()
    LOO_Z = LeaveOneOut(self.exog_nonparametric).__iter__()
    Xb = np.dot(self.exog, b)[:, None]
    L = 0
    for ii, X_not_i in enumerate(LOO_X):
        Y = next(LOO_Y)
        Z = next(LOO_Z)
        Xb_j = np.dot(X_not_i, b)[:, None]
        Yx = Y - Xb_j
        G = self.func(bw, endog=Yx, exog=-Z, data_predict=-self.exog_nonparametric[ii, :])[0]
        lt = Xb[ii, :]
        L += (self.endog[ii] - lt - G) ** 2
    return L