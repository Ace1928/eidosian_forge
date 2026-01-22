from collections.abc import Iterable
import copy  # check if needed when dropping python 2.7
import numpy as np
from scipy import optimize
import pandas as pd
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import (
import statsmodels.regression.linear_model as lm
from statsmodels.tools.sm_exceptions import (PerfectSeparationError,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
def select_penweight(self, criterion='aic', start_params=None, start_model_params=None, method='basinhopping', **fit_kwds):
    """find alpha by minimizing results criterion

        The objective for the minimization can be results attributes like
        ``gcv``, ``aic`` or ``bic`` where the latter are based on effective
        degrees of freedom.

        Warning: In many case the optimization might converge to a local
        optimum or near optimum. Different start_params or using a global
        optimizer is recommended, default is basinhopping.

        Parameters
        ----------
        criterion='aic'
            name of results attribute to be minimized.
            Default is 'aic', other options are 'gcv', 'cv' or 'bic'.
        start_params : None or array
            starting parameters for alpha in the penalization weight
            minimization. The parameters are internally exponentiated and
            the minimization is with respect to ``exp(alpha)``
        start_model_params : None or array
            starting parameter for the ``model._fit_pirls``.
        method : 'basinhopping', 'nm' or 'minimize'
            'basinhopping' and 'nm' directly use the underlying scipy.optimize
            functions `basinhopping` and `fmin`. 'minimize' provides access
            to the high level interface, `scipy.optimize.minimize`.
        fit_kwds : keyword arguments
            additional keyword arguments will be used in the call to the
            scipy optimizer. Which keywords are supported depends on the
            scipy optimization function.

        Returns
        -------
        alpha : ndarray
            penalization parameter found by minimizing the criterion.
            Note that this can be only a local (near) optimum.
        fit_res : tuple
            results returned by the scipy optimization routine. The
            parameters in the optimization problem are `log(alpha)`
        history : dict
            history of calls to pirls and contains alpha, the fit
            criterion and the parameters to which pirls converged to for the
            given alpha.

        Notes
        -----
        In the test cases Nelder-Mead and bfgs often converge to local optima,
        see also https://github.com/statsmodels/statsmodels/issues/5381.

        This does not use any analytical derivatives for the criterion
        minimization.

        Status: experimental, It is possible that defaults change if there
        is a better way to find a global optimum. API (e.g. type of return)
        might also change.
        """
    scale_keep = self.scale
    scaletype_keep = self.scaletype
    alpha_keep = copy.copy(self.alpha)
    if start_params is None:
        start_params = np.zeros(self.k_smooths)
    else:
        start_params = np.log(1e-20 + start_params)
    history = {}
    history['alpha'] = []
    history['params'] = [start_model_params]
    history['criterion'] = []

    def fun(p):
        a = np.exp(p)
        res_ = self._fit_pirls(start_params=history['params'][-1], alpha=a)
        history['alpha'].append(a)
        history['params'].append(np.asarray(res_.params))
        return getattr(res_, criterion)
    if method == 'nm':
        kwds = dict(full_output=True, maxiter=1000, maxfun=2000)
        kwds.update(fit_kwds)
        fit_res = optimize.fmin(fun, start_params, **kwds)
        opt = fit_res[0]
    elif method == 'basinhopping':
        kwds = dict(minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 100, 'maxfev': 500}}, niter=10)
        kwds.update(fit_kwds)
        fit_res = optimize.basinhopping(fun, start_params, **kwds)
        opt = fit_res.x
    elif method == 'minimize':
        fit_res = optimize.minimize(fun, start_params, **fit_kwds)
        opt = fit_res.x
    else:
        raise ValueError('method not recognized')
    del history['params'][0]
    alpha = np.exp(opt)
    self.scale = scale_keep
    self.scaletype = scaletype_keep
    self.alpha = alpha_keep
    return (alpha, fit_res, history)