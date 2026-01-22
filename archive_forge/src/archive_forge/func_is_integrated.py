import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
@property
def is_integrated(self):
    """
        (bool) Is the model integrated.

        I.e. does it have a nonzero `diff` or `seasonal_diff`.
        """
    return self.diff > 0 or self.seasonal_diff > 0