from warnings import warn
import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.tsatools import lagmat
from .initialization import Initialization
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
@property
def initial_transition(self):
    """Initial transition matrix"""
    transition = np.zeros((self.k_states, self.k_states))
    if self.state_regression:
        start = -self._k_exog
        transition[start:, start:] = np.eye(self._k_exog)
        start = -(self._k_exog + self._k_order)
        end = -self._k_exog if self._k_exog > 0 else None
    else:
        start = -self._k_order
        end = None
    if self._k_order > 0:
        transition[start:end, start:end] = companion_matrix(self._k_order)
        if self.hamilton_representation:
            transition[start:end, start:end] = np.transpose(companion_matrix(self._k_order))
    if self._k_seasonal_diff > 0:
        seasonal_companion = companion_matrix(self.seasonal_periods).T
        seasonal_companion[0, -1] = 1
        for d in range(self._k_seasonal_diff):
            start = self._k_diff + d * self.seasonal_periods
            end = self._k_diff + (d + 1) * self.seasonal_periods
            transition[start:end, start:end] = seasonal_companion
            if d < self._k_seasonal_diff - 1:
                transition[start, end + self.seasonal_periods - 1] = 1
            transition[start, self._k_states_diff] = 1
    if self._k_diff > 0:
        idx = np.triu_indices(self._k_diff)
        transition[idx] = 1
        if self.seasonal_periods > 0:
            start = self._k_diff
            end = self._k_states_diff
            transition[:self._k_diff, start:end] = ([0] * (self.seasonal_periods - 1) + [1]) * self._k_seasonal_diff
        column = self._k_states_diff
        transition[:self._k_diff, column] = 1
    return transition