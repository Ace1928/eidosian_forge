from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def proportions_chisquare_allpairs(count, nobs, multitest_method='hs'):
    """
    Chisquare test of proportions for all pairs of k samples

    Performs a chisquare test for proportions for all pairwise comparisons.
    The alternative is two-sided

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    multitest_method : str
        This chooses the method for the multiple testing p-value correction,
        that is used as default in the results.
        It can be any method that is available in  ``multipletesting``.
        The default is Holm-Sidak 'hs'.

    Returns
    -------
    result : AllPairsResults instance
        The returned results instance has several statistics, such as p-values,
        attached, and additional methods for using a non-default
        ``multitest_method``.

    Notes
    -----
    Yates continuity correction is not available.
    """
    all_pairs = lzip(*np.triu_indices(len(count), 1))
    pvals = [proportions_chisquare(count[list(pair)], nobs[list(pair)])[1] for pair in all_pairs]
    return AllPairsResults(pvals, all_pairs, multitest_method=multitest_method)