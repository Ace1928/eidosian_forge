import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like
def correlation_sum(indicators, embedding_dim):
    """
    Calculate a correlation sum

    Useful as an estimator of a correlation integral

    Parameters
    ----------
    indicators : ndarray
        2d array of distance threshold indicators
    embedding_dim : int
        embedding dimension

    Returns
    -------
    corrsum : float
        Correlation sum
    indicators_joint
        matrix of joint-distance-threshold indicators
    """
    if not indicators.ndim == 2:
        raise ValueError('Indicators must be a matrix')
    if not indicators.shape[0] == indicators.shape[1]:
        raise ValueError('Indicator matrix must be symmetric (square)')
    if embedding_dim == 1:
        indicators_joint = indicators
    else:
        corrsum, indicators = correlation_sum(indicators, embedding_dim - 1)
        indicators_joint = indicators[1:, 1:] * indicators[:-1, :-1]
    nobs = len(indicators_joint)
    corrsum = np.mean(indicators_joint[np.triu_indices(nobs, 1)])
    return (corrsum, indicators_joint)