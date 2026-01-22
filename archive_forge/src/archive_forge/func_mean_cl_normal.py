import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs, uniquecols
from ..doctools import document
from ..exceptions import PlotnineError
from .stat import stat
def mean_cl_normal(series, confidence_interval=0.95):
    """
    Mean with confidence interval assuming normal distribution

    Credit: from http://stackoverflow.com/a/15034143

    Parameters
    ----------
    series : pandas.Series
        Values
    confidence_interval : float
        Confidence interval in the range (0, 1).
    """
    import scipy.stats as stats
    a = np.asarray(series)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t._ppf((1 + confidence_interval) / 2, len(a) - 1)
    return pd.DataFrame({'y': [m], 'ymin': m - h, 'ymax': m + h})