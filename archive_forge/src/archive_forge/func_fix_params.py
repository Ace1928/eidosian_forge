from statsmodels.compat.pandas import deprecate_kwarg
import contextlib
from typing import Any
from collections.abc import Hashable, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, least_squares, minimize
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.tools.validation import (
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import (
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import HoltWintersArgs
from statsmodels.tsa.holtwinters._smoothers import (
from statsmodels.tsa.holtwinters.results import (
from statsmodels.tsa.tsatools import freq_to_period
@contextlib.contextmanager
def fix_params(self, values):
    """
        Temporarily fix parameters for estimation.

        Parameters
        ----------
        values : dict
            Values to fix. The key is the parameter name and the value is the
            fixed value.

        Yields
        ------
        None
            No value returned.

        Examples
        --------
        >>> from statsmodels.datasets.macrodata import load_pandas
        >>> data = load_pandas()
        >>> import statsmodels.tsa.api as tsa
        >>> mod = tsa.ExponentialSmoothing(data.data.realcons, trend="add",
        ...                                initialization_method="estimated")
        >>> with mod.fix_params({"smoothing_level": 0.2}):
        ...     mod.fit()
        """
    values = dict_like(values, 'values')
    valid_keys = ('smoothing_level',)
    if self.has_trend:
        valid_keys += ('smoothing_trend',)
    if self.has_seasonal:
        valid_keys += ('smoothing_seasonal',)
        m = self.seasonal_periods
        valid_keys += tuple([f'initial_seasonal.{i}' for i in range(m)])
    if self.damped_trend:
        valid_keys += ('damping_trend',)
    if self._initialization_method in ('estimated', None):
        extra_keys = [key.replace('smoothing_', 'initial_') for key in valid_keys if 'smoothing_' in key]
        valid_keys += tuple(extra_keys)
    for key in values:
        if key not in valid_keys:
            valid = ', '.join(valid_keys[:-1]) + ', and ' + valid_keys[-1]
            raise KeyError(f'{key} if not allowed. Only {valid} are supported in this specification.')
    if 'smoothing_level' in values:
        alpha = values['smoothing_level']
        if alpha <= 0.0:
            raise ValueError('smoothing_level must be in (0, 1)')
        beta = values.get('smoothing_trend', 0.0)
        if beta > alpha:
            raise ValueError('smoothing_trend must be <= smoothing_level')
        gamma = values.get('smoothing_seasonal', 0.0)
        if gamma > 1 - alpha:
            raise ValueError('smoothing_seasonal must be <= 1 - smoothing_level')
    try:
        self._fixed_parameters = values
        yield
    finally:
        self._fixed_parameters = {}