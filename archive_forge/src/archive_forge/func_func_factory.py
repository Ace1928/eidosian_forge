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
def func_factory(count: int, nobs: int) -> Callable[[float], float]:
    if hasattr(stats, 'binomtest'):

        def func(qi):
            return stats.binomtest(count, nobs, p=qi).pvalue - alpha
    else:

        def func(qi):
            return stats.binom_test(count, nobs, p=qi) - alpha
    return func