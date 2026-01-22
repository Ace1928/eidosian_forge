from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def in_sample(self, index: pd.Index) -> pd.DataFrame:
    nobs = index.shape[0]
    terms = np.empty((index.shape[0], 12))
    for i in range(0, 12, 2):
        if i == 0:
            value = 1
        elif i == 2:
            value = np.arange(nobs)
        elif i == 4:
            value = np.random.standard_normal(nobs)
        elif i == 6:
            value = np.zeros(nobs)
            value[::2] = 1
        elif i == 8:
            value = 0
        else:
            value = np.zeros(nobs)
            value[1::2] = 1
        terms[:, i] = terms[:, i + 1] = value
    return pd.DataFrame(terms, columns=self.columns, index=index)