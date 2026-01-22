import numpy as np
from scipy import stats  #get rid of this? need only norm.sf
def cohens_kappa(table, weights=None, return_results=True, wt=None):
    """Compute Cohen's kappa with variance and equal-zero test

    Parameters
    ----------
    table : array_like, 2-Dim
        square array with results of two raters, one rater in rows, second
        rater in columns
    weights : array_like
        The interpretation of weights depends on the wt argument.
        If both are None, then the simple kappa is computed.
        see wt for the case when wt is not None
        If weights is two dimensional, then it is directly used as a weight
        matrix. For computing the variance of kappa, the maximum of the
        weights is assumed to be smaller or equal to one.
        TODO: fix conflicting definitions in the 2-Dim case for
    wt : {None, str}
        If wt and weights are None, then the simple kappa is computed.
        If wt is given, but weights is None, then the weights are set to
        be [0, 1, 2, ..., k].
        If weights is a one-dimensional array, then it is used to construct
        the weight matrix given the following options.

        wt in ['linear', 'ca' or None] : use linear weights, Cicchetti-Allison
            actual weights are linear in the score "weights" difference
        wt in ['quadratic', 'fc'] : use linear weights, Fleiss-Cohen
            actual weights are squared in the score "weights" difference
        wt = 'toeplitz' : weight matrix is constructed as a toeplitz matrix
            from the one dimensional weights.

    return_results : bool
        If True (default), then an instance of KappaResults is returned.
        If False, then only kappa is computed and returned.

    Returns
    -------
    results or kappa
        If return_results is True (default), then a results instance with all
        statistics is returned
        If return_results is False, then only kappa is calculated and returned.

    Notes
    -----
    There are two conflicting definitions of the weight matrix, Wikipedia
    versus SAS manual. However, the computation are invariant to rescaling
    of the weights matrix, so there is no difference in the results.

    Weights for 'linear' and 'quadratic' are interpreted as scores for the
    categories, the weights in the computation are based on the pairwise
    difference between the scores.
    Weights for 'toeplitz' are a interpreted as weighted distance. The distance
    only depends on how many levels apart two entries in the table are but
    not on the levels themselves.

    example:

    weights = '0, 1, 2, 3' and wt is either linear or toeplitz means that the
    weighting only depends on the simple distance of levels.

    weights = '0, 0, 1, 1' and wt = 'linear' means that the first two levels
    are zero distance apart and the same for the last two levels. This is
    the sample as forming two aggregated levels by merging the first two and
    the last two levels, respectively.

    weights = [0, 1, 2, 3] and wt = 'quadratic' is the same as squaring these
    weights and using wt = 'toeplitz'.

    References
    ----------
    Wikipedia
    SAS Manual

    """
    table = np.asarray(table, float)
    agree = np.diag(table).sum()
    nobs = table.sum()
    probs = table / nobs
    freqs = probs
    probs_diag = np.diag(probs)
    freq_row = table.sum(1) / nobs
    freq_col = table.sum(0) / nobs
    prob_exp = freq_col * freq_row[:, None]
    assert np.allclose(prob_exp.sum(), 1)
    agree_exp = np.diag(prob_exp).sum()
    if weights is None and wt is None:
        kind = 'Simple'
        kappa = (agree / nobs - agree_exp) / (1 - agree_exp)
        if return_results:
            term_a = probs_diag * (1 - (freq_row + freq_col) * (1 - kappa)) ** 2
            term_a = term_a.sum()
            term_b = probs * (freq_col[:, None] + freq_row) ** 2
            d_idx = np.arange(table.shape[0])
            term_b[d_idx, d_idx] = 0
            term_b = (1 - kappa) ** 2 * term_b.sum()
            term_c = (kappa - agree_exp * (1 - kappa)) ** 2
            var_kappa = (term_a + term_b - term_c) / (1 - agree_exp) ** 2 / nobs
            term_c = freq_col * freq_row * (freq_col + freq_row)
            var_kappa0 = agree_exp + agree_exp ** 2 - term_c.sum()
            var_kappa0 /= (1 - agree_exp) ** 2 * nobs
    else:
        if weights is None:
            weights = np.arange(table.shape[0])
        kind = 'Weighted'
        weights = np.asarray(weights, float)
        if weights.ndim == 1:
            if wt in ['ca', 'linear', None]:
                weights = np.abs(weights[:, None] - weights) / (weights[-1] - weights[0])
            elif wt in ['fc', 'quadratic']:
                weights = (weights[:, None] - weights) ** 2 / (weights[-1] - weights[0]) ** 2
            elif wt == 'toeplitz':
                from scipy.linalg import toeplitz
                weights = toeplitz(weights)
            else:
                raise ValueError('wt option is not known')
        else:
            rows, cols = table.shape
            if table.shape != weights.shape:
                raise ValueError('weights are not square')
        kappa = 1 - (weights * table).sum() / nobs / (weights * prob_exp).sum()
        if return_results:
            var_kappa = np.nan
            var_kappa0 = np.nan
            w = 1.0 - weights
            w_row = (freq_col * w).sum(1)
            w_col = (freq_row[:, None] * w).sum(0)
            agree_wexp = (w * freq_col * freq_row[:, None]).sum()
            term_a = freqs * (w - (w_col + w_row[:, None]) * (1 - kappa)) ** 2
            fac = 1.0 / ((1 - agree_wexp) ** 2 * nobs)
            var_kappa = term_a.sum() - (kappa - agree_wexp * (1 - kappa)) ** 2
            var_kappa *= fac
            freqse = freq_col * freq_row[:, None]
            var_kappa0 = (freqse * (w - (w_col + w_row[:, None])) ** 2).sum()
            var_kappa0 -= agree_wexp ** 2
            var_kappa0 *= fac
    kappa_max = (np.minimum(freq_row, freq_col).sum() - agree_exp) / (1 - agree_exp)
    if return_results:
        res = KappaResults(kind=kind, kappa=kappa, kappa_max=kappa_max, weights=weights, var_kappa=var_kappa, var_kappa0=var_kappa0)
        return res
    else:
        return kappa