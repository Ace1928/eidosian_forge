import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
def fit_regularized(self, method='elastic_net', alpha=0.0, start_params=None, refit=False, **kwargs):
    """
        Return a regularized fit to a linear regression model.

        Parameters
        ----------
        method : {'elastic_net'}
            Only the `elastic_net` approach is currently implemented.
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        start_params : array_like
            Starting values for `params`.
        refit : bool
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        **kwargs
            Additional keyword arguments used to fit the model.

        Returns
        -------
        PHRegResults
            Returns a results instance.

        Notes
        -----
        The penalty is the ``elastic net`` penalty, which is a
        combination of L1 and L2 penalties.

        The function that is minimized is:

        .. math::

            -loglike/n + alpha*((1-L1\\_wt)*|params|_2^2/2 + L1\\_wt*|params|_1)

        where :math:`|*|_1` and :math:`|*|_2` are the L1 and L2 norms.

        Post-estimation results are based on the same data used to
        select variables, hence may be subject to overfitting biases.

        The elastic_net method uses the following keyword arguments:

        maxiter : int
            Maximum number of iterations
        L1_wt  : float
            Must be in [0, 1].  The L1 penalty has weight L1_wt and the
            L2 penalty has weight 1 - L1_wt.
        cnvrg_tol : float
            Convergence threshold for line searches
        zero_tol : float
            Coefficients below this threshold are treated as zero.
        """
    from statsmodels.base.elastic_net import fit_elasticnet
    if method != 'elastic_net':
        raise ValueError('method for fit_regularized must be elastic_net')
    defaults = {'maxiter': 50, 'L1_wt': 1, 'cnvrg_tol': 1e-10, 'zero_tol': 1e-10}
    defaults.update(kwargs)
    return fit_elasticnet(self, method=method, alpha=alpha, start_params=start_params, refit=refit, **defaults)