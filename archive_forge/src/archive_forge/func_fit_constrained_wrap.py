import numpy as np
def fit_constrained_wrap(model, constraints, start_params=None, **fit_kwds):
    """fit_constraint that returns a results instance

    This is a development version for fit_constrained methods or
    fit_constrained as standalone function.

    It will not work correctly for all models because creating a new
    results instance is not standardized for use outside the `fit` methods,
    and might need adjustements for this.

    This is the prototype for the fit_constrained method that has been added
    to Poisson and GLM.
    """
    self = model
    from patsy import DesignInfo
    lc = DesignInfo(self.exog_names).linear_constraint(constraints)
    R, q = (lc.coefs, lc.constants)
    params, cov, res_constr = fit_constrained(self, R, q, start_params=start_params, fit_kwds=fit_kwds)
    res = self.fit(start_params=params, maxiter=0, warn_convergence=False)
    res._results.params = params
    res._results.cov_params_default = cov
    cov_type = fit_kwds.get('cov_type', 'nonrobust')
    if cov_type == 'nonrobust':
        res._results.normalized_cov_params = cov / res_constr.scale
    else:
        res._results.normalized_cov_params = None
    k_constr = len(q)
    res._results.df_resid += k_constr
    res._results.df_model -= k_constr
    res._results.constraints = LinearConstraints.from_patsy(lc)
    res._results.k_constr = k_constr
    res._results.results_constrained = res_constr
    return res