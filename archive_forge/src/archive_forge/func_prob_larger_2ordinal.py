import numpy as np
from scipy import stats
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.effect_size import _noncentrality_chisquare
def prob_larger_2ordinal(probs1, probs2):
    """Stochastically large probability for two ordinal distributions

    Computes Pr(x1 > x2) + 0.5 * Pr(x1 = x2) for two ordered multinomial
    (ordinal) distributed random variables x1 and x2.

    This is vectorized with choices along last axis.
    Broadcasting if freq2 is 1-dim also seems to work correctly.

    Returns
    -------
    prob1 : float
        Probability that random draw from distribution 1 is larger than a
        random draw from distribution 2. Pr(x1 > x2) + 0.5 * Pr(x1 = x2)
    prob2 : float
        prob2 = 1 - prob1 = Pr(x1 < x2) + 0.5 * Pr(x1 = x2)
    """
    freq1 = np.asarray(probs1)
    freq2 = np.asarray(probs2)
    freq1_ = np.concatenate((np.zeros(freq1.shape[:-1] + (1,)), freq1), axis=-1)
    freq2_ = np.concatenate((np.zeros(freq2.shape[:-1] + (1,)), freq2), axis=-1)
    cdf1 = freq1_.cumsum(axis=-1)
    cdf2 = freq2_.cumsum(axis=-1)
    cdfm1 = (cdf1[..., 1:] + cdf1[..., :-1]) / 2
    cdfm2 = (cdf2[..., 1:] + cdf2[..., :-1]) / 2
    prob1 = (cdfm2 * freq1).sum(-1)
    prob2 = (cdfm1 * freq2).sum(-1)
    return (prob1, prob2)