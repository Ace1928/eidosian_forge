from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
def get_corr_func(method: CorrelationMethod) -> Callable[[np.ndarray, np.ndarray], float]:
    if method == 'kendall':
        from scipy.stats import kendalltau

        def func(a, b):
            return kendalltau(a, b)[0]
        return func
    elif method == 'spearman':
        from scipy.stats import spearmanr

        def func(a, b):
            return spearmanr(a, b)[0]
        return func
    elif method == 'pearson':

        def func(a, b):
            return np.corrcoef(a, b)[0, 1]
        return func
    elif callable(method):
        return method
    raise ValueError(f"Unknown method '{method}', expected one of 'kendall', 'spearman', 'pearson', or callable")