import numpy as np
from statsmodels.base.model import Results
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
def fit_elasticnet(model, method='coord_descent', maxiter=100, alpha=0.0, L1_wt=1.0, start_params=None, cnvrg_tol=1e-07, zero_tol=1e-08, refit=False, check_step=True, loglike_kwds=None, score_kwds=None, hess_kwds=None):
    """
    Return an elastic net regularized fit to a regression model.

    Parameters
    ----------
    model : model object
        A statsmodels object implementing ``loglike``, ``score``, and
        ``hessian``.
    method : {'coord_descent'}
        Only the coordinate descent algorithm is implemented.
    maxiter : int
        The maximum number of iteration cycles (an iteration cycle
        involves running coordinate descent on all variables).
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.
    L1_wt : scalar
        The fraction of the penalty given to the L1 penalty term.
        Must be between 0 and 1 (inclusive).  If 0, the fit is
        a ridge fit, if 1 it is a lasso fit.
    start_params : array_like
        Starting values for `params`.
    cnvrg_tol : scalar
        If `params` changes by less than this amount (in sup-norm)
        in one iteration cycle, the algorithm terminates with
        convergence.
    zero_tol : scalar
        Any estimated coefficient smaller than this value is
        replaced with zero.
    refit : bool
        If True, the model is refit using only the variables that have
        non-zero coefficients in the regularized fit.  The refitted
        model is not regularized.
    check_step : bool
        If True, confirm that the first step is an improvement and search
        further if it is not.
    loglike_kwds : dict-like or None
        Keyword arguments for the log-likelihood function.
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    hess_kwds : dict-like or None
        Keyword arguments for the Hessian function.

    Returns
    -------
    Results
        A results object.

    Notes
    -----
    The ``elastic net`` penalty is a combination of L1 and L2
    penalties.

    The function that is minimized is:

    -loglike/n + alpha*((1-L1_wt)*|params|_2^2/2 + L1_wt*|params|_1)

    where |*|_1 and |*|_2 are the L1 and L2 norms.

    The computational approach used here is to obtain a quadratic
    approximation to the smooth part of the target function:

    -loglike/n + alpha*(1-L1_wt)*|params|_2^2/2

    then repeatedly optimize the L1 penalized version of this function
    along coordinate axes.
    """
    k_exog = model.exog.shape[1]
    loglike_kwds = {} if loglike_kwds is None else loglike_kwds
    score_kwds = {} if score_kwds is None else score_kwds
    hess_kwds = {} if hess_kwds is None else hess_kwds
    if np.isscalar(alpha):
        alpha = alpha * np.ones(k_exog)
    if start_params is None:
        params = np.zeros(k_exog)
    else:
        params = start_params.copy()
    btol = 0.0001
    params_zero = np.zeros(len(params), dtype=bool)
    init_args = model._get_init_kwds()
    init_args['hasconst'] = False
    model_offset = init_args.pop('offset', None)
    if 'exposure' in init_args and init_args['exposure'] is not None:
        if model_offset is None:
            model_offset = np.log(init_args.pop('exposure'))
        else:
            model_offset += np.log(init_args.pop('exposure'))
    fgh_list = [_gen_npfuncs(k, L1_wt, alpha, loglike_kwds, score_kwds, hess_kwds) for k in range(k_exog)]
    converged = False
    for itr in range(maxiter):
        params_save = params.copy()
        for k in range(k_exog):
            if params_zero[k]:
                continue
            params0 = params.copy()
            params0[k] = 0
            offset = np.dot(model.exog, params0)
            if model_offset is not None:
                offset += model_offset
            model_1var = model.__class__(model.endog, model.exog[:, k], offset=offset, **init_args)
            func, grad, hess = fgh_list[k]
            params[k] = _opt_1d(func, grad, hess, model_1var, params[k], alpha[k] * L1_wt, tol=btol, check_step=check_step)
            if itr > 0 and np.abs(params[k]) < zero_tol:
                params_zero[k] = True
                params[k] = 0.0
        pchange = np.max(np.abs(params - params_save))
        if pchange < cnvrg_tol:
            converged = True
            break
    params[np.abs(params) < zero_tol] = 0
    if not refit:
        results = RegularizedResults(model, params)
        results.converged = converged
        return RegularizedResultsWrapper(results)
    ii = np.flatnonzero(params)
    cov = np.zeros((k_exog, k_exog))
    init_args = {k: getattr(model, k, None) for k in model._init_keys}
    if len(ii) > 0:
        model1 = model.__class__(model.endog, model.exog[:, ii], **init_args)
        rslt = model1.fit()
        params[ii] = rslt.params
        cov[np.ix_(ii, ii)] = rslt.normalized_cov_params
    else:
        model1 = model.__class__(model.endog, model.exog[:, 0], **init_args)
        rslt = model1.fit(maxiter=0)
    if issubclass(rslt.__class__, wrap.ResultsWrapper):
        klass = rslt._results.__class__
    else:
        klass = rslt.__class__
    if hasattr(rslt, 'scale'):
        scale = rslt.scale
    else:
        scale = 1.0
    p, q = (model.df_model, model.df_resid)
    model.df_model = len(ii)
    model.df_resid = model.nobs - model.df_model
    refit = klass(model, params, cov, scale=scale)
    refit.regularized = True
    refit.converged = converged
    refit.method = method
    refit.fit_history = {'iteration': itr + 1}
    model.df_model, model.df_resid = (p, q)
    return refit