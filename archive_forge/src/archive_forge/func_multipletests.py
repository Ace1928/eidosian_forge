import numpy as np
from statsmodels.stats._knockoff import RegressionFDR
def multipletests(pvals, alpha=0.05, method='hs', maxiter=1, is_sorted=False, returnsorted=False):
    """
    Test results and p-value correction for multiple tests

    Parameters
    ----------
    pvals : array_like, 1-d
        uncorrected p-values.   Must be 1-dimensional.
    alpha : float
        FWER, family-wise error rate, e.g. 0.1
    method : str
        Method used for testing and adjustment of pvalues. Can be either the
        full name or initial letters. Available methods are:

        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` : step down method using Sidak adjustments
        - `holm` : step-down method using Bonferroni adjustments
        - `simes-hochberg` : step-up method  (independent)
        - `hommel` : closed method based on Simes tests (non-negative)
        - `fdr_bh` : Benjamini/Hochberg  (non-negative)
        - `fdr_by` : Benjamini/Yekutieli (negative)
        - `fdr_tsbh` : two stage fdr correction (non-negative)
        - `fdr_tsbky` : two stage fdr correction (non-negative)

    maxiter : int or bool
        Maximum number of iterations for two-stage fdr, `fdr_tsbh` and
        `fdr_tsbky`. It is ignored by all other methods.
        maxiter=1 (default) corresponds to the two stage method.
        maxiter=-1 corresponds to full iterations which is maxiter=len(pvals).
        maxiter=0 uses only a single stage fdr correction using a 'bh' or 'bky'
        prior fraction of assumed true hypotheses.
    is_sorted : bool
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.
    returnsorted : bool
         not tested, return sorted p-values instead of original sequence

    Returns
    -------
    reject : ndarray, boolean
        true for hypothesis that can be rejected for given alpha
    pvals_corrected : ndarray
        p-values corrected for multiple tests
    alphacSidak : float
        corrected alpha for Sidak method
    alphacBonf : float
        corrected alpha for Bonferroni method

    Notes
    -----
    There may be API changes for this function in the future.

    Except for 'fdr_twostage', the p-value correction is independent of the
    alpha specified as argument. In these cases the corrected p-values
    can also be compared with a different alpha. In the case of 'fdr_twostage',
    the corrected p-values are specific to the given alpha, see
    ``fdrcorrection_twostage``.

    The 'fdr_gbs' procedure is not verified against another package, p-values
    are derived from scratch and are not derived in the reference. In Monte
    Carlo experiments the method worked correctly and maintained the false
    discovery rate.

    All procedures that are included, control FWER or FDR in the independent
    case, and most are robust in the positively correlated case.

    `fdr_gbs`: high power, fdr control for independent case and only small
    violation in positively correlated case

    **Timing**:

    Most of the time with large arrays is spent in `argsort`. When
    we want to calculate the p-value for several methods, then it is more
    efficient to presort the pvalues, and put the results back into the
    original order outside of the function.

    Method='hommel' is very slow for large arrays, since it requires the
    evaluation of n partitions, where n is the number of p-values.
    """
    import gc
    pvals = np.asarray(pvals)
    alphaf = alpha
    if not is_sorted:
        sortind = np.argsort(pvals)
        pvals = np.take(pvals, sortind)
    ntests = len(pvals)
    alphacSidak = 1 - np.power(1.0 - alphaf, 1.0 / ntests)
    alphacBonf = alphaf / float(ntests)
    if method.lower() in ['b', 'bonf', 'bonferroni']:
        reject = pvals <= alphacBonf
        pvals_corrected = pvals * float(ntests)
    elif method.lower() in ['s', 'sidak']:
        reject = pvals <= alphacSidak
        pvals_corrected = -np.expm1(ntests * np.log1p(-pvals))
    elif method.lower() in ['hs', 'holm-sidak']:
        alphacSidak_all = 1 - np.power(1.0 - alphaf, 1.0 / np.arange(ntests, 0, -1))
        notreject = pvals > alphacSidak_all
        del alphacSidak_all
        nr_index = np.nonzero(notreject)[0]
        if nr_index.size == 0:
            notrejectmin = len(pvals)
        else:
            notrejectmin = np.min(nr_index)
        notreject[notrejectmin:] = True
        reject = ~notreject
        del notreject
        pvals_corrected_raw = -np.expm1(np.arange(ntests, 0, -1) * np.log1p(-pvals))
        pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)
        del pvals_corrected_raw
    elif method.lower() in ['h', 'holm']:
        notreject = pvals > alphaf / np.arange(ntests, 0, -1)
        nr_index = np.nonzero(notreject)[0]
        if nr_index.size == 0:
            notrejectmin = len(pvals)
        else:
            notrejectmin = np.min(nr_index)
        notreject[notrejectmin:] = True
        reject = ~notreject
        pvals_corrected_raw = pvals * np.arange(ntests, 0, -1)
        pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)
        del pvals_corrected_raw
        gc.collect()
    elif method.lower() in ['sh', 'simes-hochberg']:
        alphash = alphaf / np.arange(ntests, 0, -1)
        reject = pvals <= alphash
        rejind = np.nonzero(reject)
        if rejind[0].size > 0:
            rejectmax = np.max(np.nonzero(reject))
            reject[:rejectmax] = True
        pvals_corrected_raw = np.arange(ntests, 0, -1) * pvals
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        del pvals_corrected_raw
    elif method.lower() in ['ho', 'hommel']:
        a = pvals.copy()
        for m in range(ntests, 1, -1):
            cim = np.min(m * pvals[-m:] / np.arange(1, m + 1.0))
            a[-m:] = np.maximum(a[-m:], cim)
            a[:-m] = np.maximum(a[:-m], np.minimum(m * pvals[:-m], cim))
        pvals_corrected = a
        reject = a <= alphaf
    elif method.lower() in ['fdr_bh', 'fdr_i', 'fdr_p', 'fdri', 'fdrp']:
        reject, pvals_corrected = fdrcorrection(pvals, alpha=alpha, method='indep', is_sorted=True)
    elif method.lower() in ['fdr_by', 'fdr_n', 'fdr_c', 'fdrn', 'fdrcorr']:
        reject, pvals_corrected = fdrcorrection(pvals, alpha=alpha, method='n', is_sorted=True)
    elif method.lower() in ['fdr_tsbky', 'fdr_2sbky', 'fdr_twostage']:
        reject, pvals_corrected = fdrcorrection_twostage(pvals, alpha=alpha, method='bky', maxiter=maxiter, is_sorted=True)[:2]
    elif method.lower() in ['fdr_tsbh', 'fdr_2sbh']:
        reject, pvals_corrected = fdrcorrection_twostage(pvals, alpha=alpha, method='bh', maxiter=maxiter, is_sorted=True)[:2]
    elif method.lower() in ['fdr_gbs']:
        ii = np.arange(1, ntests + 1)
        q = (ntests + 1.0 - ii) / ii * pvals / (1.0 - pvals)
        pvals_corrected_raw = np.maximum.accumulate(q)
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        del pvals_corrected_raw
        reject = pvals_corrected <= alpha
    else:
        raise ValueError('method not recognized')
    if pvals_corrected is not None:
        pvals_corrected[pvals_corrected > 1] = 1
    if is_sorted or returnsorted:
        return (reject, pvals_corrected, alphacSidak, alphacBonf)
    else:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[sortind] = reject
        return (reject_, pvals_corrected_, alphacSidak, alphacBonf)