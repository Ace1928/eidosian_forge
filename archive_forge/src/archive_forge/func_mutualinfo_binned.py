import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import statsmodels.sandbox.infotheo as infotheo
def mutualinfo_binned(y, x, bins, normed=True):
    """mutual information of two random variables estimated with kde



    Notes
    -----
    bins='auto' selects the number of bins so that approximately 5 observations
    are expected to be in each bin under the assumption of independence. This
    follows roughly the description in Kahn et al. 2007

    """
    nobs = len(x)
    if not len(y) == nobs:
        raise ValueError('both data arrays need to have the same size')
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if bins == 'auto':
        ys = np.sort(y)
        xs = np.sort(x)
        qbin_sqr = np.sqrt(5.0 / nobs)
        quantiles = np.linspace(0, 1, 1.0 / qbin_sqr)
        quantile_index = ((nobs - 1) * quantiles).astype(int)
        shift = 1e-06 + np.ones(quantiles.shape)
        shift[0] -= 2 * 1e-06
        binsy = ys[quantile_index] + shift
        binsx = xs[quantile_index] + shift
    elif np.size(bins) == 1:
        binsy = bins
        binsx = bins
    elif len(bins) == 2:
        binsy, binsx = bins
    fx, binsx = np.histogram(x, bins=binsx)
    fy, binsy = np.histogram(y, bins=binsy)
    fyx, binsy, binsx = np.histogram2d(y, x, bins=(binsy, binsx))
    pyx = fyx * 1.0 / nobs
    px = fx * 1.0 / nobs
    py = fy * 1.0 / nobs
    mi_obs = pyx * (np.log(pyx + 1e-10) - np.log(py)[:, None] - np.log(px))
    mi = mi_obs.sum()
    if normed:
        mi_normed = np.sqrt(1.0 - np.exp(-2 * mi))
        return (mi_normed, (pyx, py, px, binsy, binsx), mi_obs)
    else:
        return mi