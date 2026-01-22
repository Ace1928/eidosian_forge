import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def cov_nw_groupsum(results, nlags, time, weights_func=weights_bartlett, use_correction=0):
    """Driscoll and Kraay Panel robust covariance matrix

    Robust covariance matrix for panel data of Driscoll and Kraay.

    Assumes we have a panel of time series where the time index is available.
    The time index is assumed to represent equal spaced periods. At least one
    observation per period is required.

    Parameters
    ----------
    results : result instance
       result of a regression, uses results.model.exog and results.resid
       TODO: this should use wexog instead
    nlags : int or None
        Highest lag to include in kernel window. Currently, no default
        because the optimal length will depend on the number of observations
        per cross-sectional unit.
    time : ndarray of int
        this should contain the coding for the time period of each observation.
        time periods should be integers in range(maxT) where maxT is obs of i
    weights_func : callable
        weights_func is called with nlags as argument to get the kernel
        weights. default are Bartlett weights
    use_correction : 'cluster' or 'hac' or False
        If False, then no small sample correction is used.
        If 'hac' (default), then the same correction as in single time series, cov_hac
        is used.
        If 'cluster', then the same correction as in cov_cluster is
        used.

    Returns
    -------
    cov : ndarray, (k_vars, k_vars)
        HAC robust covariance matrix for parameter estimates

    Notes
    -----
    Tested against STATA xtscc package, which uses no small sample correction

    This first averages relevant variables for each time period over all
    individuals/groups, and then applies the same kernel weighted averaging
    over time as in HAC.

    Warning:
    In the example with a short panel (few time periods and many individuals)
    with mainly across individual variation this estimator did not produce
    reasonable results.

    Options might change when other kernels besides Bartlett and uniform are
    available.

    References
    ----------
    Daniel Hoechle, xtscc paper
    Driscoll and Kraay

    """
    xu, hessian_inv = _get_sandwich_arrays(results)
    S_hac = S_hac_groupsum(xu, time, nlags=nlags, weights_func=weights_func)
    cov_hac = _HCCM2(hessian_inv, S_hac)
    if use_correction:
        nobs, k_params = xu.shape
        if use_correction == 'hac':
            cov_hac *= nobs / float(nobs - k_params)
        elif use_correction in ['c', 'cluster']:
            n_groups = len(np.unique(time))
            cov_hac *= n_groups / (n_groups - 1.0)
            cov_hac *= (nobs - 1.0) / float(nobs - k_params)
    return cov_hac