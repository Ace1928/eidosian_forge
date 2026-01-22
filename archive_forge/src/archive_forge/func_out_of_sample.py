from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def out_of_sample(self, steps: int, index: pd.Index, forecast_index: pd.Index=None) -> pd.DataFrame:
    fcast_index = self._extend_index(index, steps, forecast_index)
    terms = np.random.standard_normal((steps, 12))
    return pd.DataFrame(terms, columns=self.columns, index=fcast_index)