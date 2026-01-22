from __future__ import annotations
from statsmodels.compat.python import lrange
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import Literal
from statsmodels.tools.data import _is_recarray, _is_using_pandas
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.typing import NDArray
from statsmodels.tools.validation import (
def unvech(v):
    rows = 0.5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))
    result = np.zeros((rows, rows))
    result[np.triu_indices(rows)] = v
    result = result + result.T
    result[np.diag_indices(rows)] /= 2
    return result