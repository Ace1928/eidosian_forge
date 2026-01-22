import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def cov_nw_panel(results, nlags, groupidx, weights_func=weights_bartlett, use_correction='hac'):
    """Panel HAC robust covariance matrix

    Assumes we have a panel of time series with consecutive, equal spaced time
    periods. Data is assumed to be in long format with time series of each
    individual stacked into one array. Panel can be unbalanced.

    Parameters
    ----------
    results : result instance
       result of a regression, uses results.model.exog and results.resid
       TODO: this should use wexog instead
    nlags : int or None
        Highest lag to include in kernel window. Currently, no default
        because the optimal length will depend on the number of observations
        per cross-sectional unit.
    groupidx : list of tuple
        each tuple should contain the start and end index for an individual.
        (groupidx might change in future).
    weights_func : callable
        weights_func is called with nlags as argument to get the kernel
        weights. default are Bartlett weights
    use_correction : 'cluster' or 'hac' or False
        If False, then no small sample correction is used.
        If 'cluster' (default), then the same correction as in cov_cluster is
        used.
        If 'hac', then the same correction as in single time series, cov_hac
        is used.


    Returns
    -------
    cov : ndarray, (k_vars, k_vars)
        HAC robust covariance matrix for parameter estimates

    Notes
    -----
    For nlags=0, this is just White covariance, cov_white.
    If kernel is uniform, `weights_uniform`, with nlags equal to the number
    of observations per unit in a balance panel, then cov_cluster and
    cov_hac_panel are identical.

    Tested against STATA `newey` command with same defaults.

    Options might change when other kernels besides Bartlett and uniform are
    available.

    """
    if nlags == 0:
        weights = [1, 0]
    else:
        weights = weights_func(nlags)
    xu, hessian_inv = _get_sandwich_arrays(results)
    S_hac = S_nw_panel(xu, weights, groupidx)
    cov_hac = _HCCM2(hessian_inv, S_hac)
    if use_correction:
        nobs, k_params = xu.shape
        if use_correction == 'hac':
            cov_hac *= nobs / float(nobs - k_params)
        elif use_correction in ['c', 'clu', 'cluster']:
            n_groups = len(groupidx)
            cov_hac *= n_groups / (n_groups - 1.0)
            cov_hac *= (nobs - 1.0) / float(nobs - k_params)
    return cov_hac