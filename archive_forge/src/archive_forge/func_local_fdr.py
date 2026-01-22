import numpy as np
from statsmodels.stats._knockoff import RegressionFDR
def local_fdr(zscores, null_proportion=1.0, null_pdf=None, deg=7, nbins=30, alpha=0):
    """
    Calculate local FDR values for a list of Z-scores.

    Parameters
    ----------
    zscores : array_like
        A vector of Z-scores
    null_proportion : float
        The assumed proportion of true null hypotheses
    null_pdf : function mapping reals to positive reals
        The density of null Z-scores; if None, use standard normal
    deg : int
        The maximum exponent in the polynomial expansion of the
        density of non-null Z-scores
    nbins : int
        The number of bins for estimating the marginal density
        of Z-scores.
    alpha : float
        Use Poisson ridge regression with parameter alpha to estimate
        the density of non-null Z-scores.

    Returns
    -------
    fdr : array_like
        A vector of FDR values

    References
    ----------
    B Efron (2008).  Microarrays, Empirical Bayes, and the Two-Groups
    Model.  Statistical Science 23:1, 1-22.

    Examples
    --------
    Basic use (the null Z-scores are taken to be standard normal):

    >>> from statsmodels.stats.multitest import local_fdr
    >>> import numpy as np
    >>> zscores = np.random.randn(30)
    >>> fdr = local_fdr(zscores)

    Use a Gaussian null distribution estimated from the data:

    >>> null = EmpiricalNull(zscores)
    >>> fdr = local_fdr(zscores, null_pdf=null.pdf)
    """
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod.generalized_linear_model import families
    from statsmodels.regression.linear_model import OLS
    minz = min(zscores)
    maxz = max(zscores)
    bins = np.linspace(minz, maxz, nbins)
    zhist = np.histogram(zscores, bins)[0]
    zbins = (bins[:-1] + bins[1:]) / 2
    dmat = np.vander(zbins, deg + 1)
    sd = dmat.std(0)
    ii = sd > 1e-08
    dmat[:, ii] /= sd[ii]
    start = OLS(np.log(1 + zhist), dmat).fit().params
    if alpha > 0:
        md = GLM(zhist, dmat, family=families.Poisson()).fit_regularized(L1_wt=0, alpha=alpha, start_params=start)
    else:
        md = GLM(zhist, dmat, family=families.Poisson()).fit(start_params=start)
    dmat_full = np.vander(zscores, deg + 1)
    dmat_full[:, ii] /= sd[ii]
    fz = md.predict(dmat_full) / (len(zscores) * (bins[1] - bins[0]))
    if null_pdf is None:
        f0 = np.exp(-0.5 * zscores ** 2) / np.sqrt(2 * np.pi)
    else:
        f0 = null_pdf(zscores)
    fdr = null_proportion * f0 / fz
    fdr = np.clip(fdr, 0, 1)
    return fdr