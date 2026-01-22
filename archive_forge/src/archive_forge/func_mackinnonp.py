from scipy.stats import norm
from numpy import array, polyval, inf, asarray
def mackinnonp(teststat, regression='c', N=1, lags=None):
    """
    Returns MacKinnon's approximate p-value for teststat.

    Parameters
    ----------
    teststat : float
        "T-value" from an Augmented Dickey-Fuller regression.
    regression : str {"c", "n", "ct", "ctt"}
        This is the method of regression that was used.  Following MacKinnon's
        notation, this can be "c" for constant, "n" for no constant, "ct" for
        constant and trend, and "ctt" for constant, trend, and trend-squared.
    N : int
        The number of series believed to be I(1).  For (Augmented) Dickey-
        Fuller N = 1.

    Returns
    -------
    p-value : float
        The p-value for the ADF statistic estimated using MacKinnon 1994.

    References
    ----------
    .. [*] MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions
        for Unit-Root and Cointegration Tests." Journal of Business & Economics
        Statistics, 12.2, 167-76.

    Notes
    -----
    For (A)DF
    H_0: AR coefficient = 1
    H_a: AR coefficient < 1
    """
    maxstat = _tau_maxs[regression]
    minstat = _tau_mins[regression]
    starstat = _tau_stars[regression]
    if teststat > maxstat[N - 1]:
        return 1.0
    elif teststat < minstat[N - 1]:
        return 0.0
    if teststat <= starstat[N - 1]:
        tau_coef = _tau_smallps[regression][N - 1]
    else:
        tau_coef = _tau_largeps[regression][N - 1]
    return norm.cdf(polyval(tau_coef[::-1], teststat))