from math import log, exp
def cdf_chi2(df, stat):
    """Compute p-value, from distribution function and test statistics."""
    if df < 1:
        raise ValueError('df must be at least 1')
    if stat < 0:
        raise ValueError('The test statistic must be positive')
    x = 0.5 * stat
    alpha = df / 2.0
    prob = 1 - _incomplete_gamma(x, alpha)
    return prob