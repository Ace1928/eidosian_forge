import numpy as np
import pandas as pd
from .._utils import resolution
from ..doctools import document
from .stat import stat
def weighted_boxplot_stats(x, weights=None, whis=1.5):
    """
    Calculate weighted boxplot plot statistics

    Parameters
    ----------
    x : array_like
        Data
    weights : array_like
        Weights associated with the data.
    whis : float
        Position of the whiskers beyond the interquartile range.
        The data beyond the whisker are considered outliers.

        If a float, the lower whisker is at the lowest datum above
        `Q1 - whis*(Q3-Q1)`, and the upper whisker at the highest
        datum below `Q3 + whis*(Q3-Q1)`, where Q1 and Q3 are the
        first and third quartiles.  The default value of
        `whis = 1.5` corresponds to Tukey's original definition of
        boxplots.

    Notes
    -----
    This method adapted from Matplotlibs boxplot_stats. The key difference
    is the use of a weighted percentile calculation and then using linear
    interpolation to map weight percentiles back to data.
    """
    if weights is None:
        q1, med, q3 = np.percentile(x, (25, 50, 75))
        n = len(x)
    else:
        q1, med, q3 = weighted_percentile(x, (25, 50, 75), weights)
        n = np.sum(weights)
    iqr = q3 - q1
    mean = np.average(x, weights=weights)
    cilo = med - 1.58 * iqr / np.sqrt(n)
    cihi = med + 1.58 * iqr / np.sqrt(n)
    loval = q1 - whis * iqr
    lox = x[x >= loval]
    whislo = q1 if len(lox) == 0 or np.min(lox) > q1 else np.min(lox)
    hival = q3 + whis * iqr
    hix = x[x <= hival]
    whishi = q3 if len(hix) == 0 or np.max(hix) < q3 else np.max(hix)
    bpstats = {'fliers': x[(x < whislo) | (x > whishi)], 'mean': mean, 'med': med, 'q1': q1, 'q3': q3, 'iqr': iqr, 'whislo': whislo, 'whishi': whishi, 'cilo': cilo, 'cihi': cihi}
    return bpstats