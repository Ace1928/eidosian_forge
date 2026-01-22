from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
def pooled_odds_ratio(self, tables):
    """
        Returns the pooled odds ratio for a list of 2x2 tables.

        The pooled odds ratio is the inverse variance weighted average
        of the sample odds ratios of the tables.
        """
    if len(tables) == 0:
        return 1.0
    log_oddsratio, var = ([], [])
    for table in tables:
        lor = np.log(table[1, 1]) + np.log(table[0, 0]) - np.log(table[0, 1]) - np.log(table[1, 0])
        log_oddsratio.append(lor)
        var.append((1 / table.astype(np.float64)).sum())
    wts = [1 / v for v in var]
    wtsum = sum(wts)
    wts = [w / wtsum for w in wts]
    log_pooled_or = sum([w * e for w, e in zip(wts, log_oddsratio)])
    return np.exp(log_pooled_or)