import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
from statsmodels.tools.validation import array_like
def runstest_1samp(x, cutoff='mean', correction=True):
    """use runs test on binary discretized data above/below cutoff

    Parameters
    ----------
    x : array_like
        data, numeric
    cutoff : {'mean', 'median'} or number
        This specifies the cutoff to split the data into large and small
        values.
    correction : bool
        Following the SAS manual, for samplesize below 50, the test
        statistic is corrected by 0.5. This can be turned off with
        correction=False, and was included to match R, tseries, which
        does not use any correction.

    Returns
    -------
    z_stat : float
        test statistic, asymptotically normally distributed
    p-value : float
        p-value, reject the null hypothesis if it is below an type 1 error
        level, alpha .

    """
    x = array_like(x, 'x')
    if cutoff == 'mean':
        cutoff = np.mean(x)
    elif cutoff == 'median':
        cutoff = np.median(x)
    else:
        cutoff = float(cutoff)
    xindicator = (x >= cutoff).astype(int)
    return Runs(xindicator).runs_test(correction=correction)