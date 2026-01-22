from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
def numeric_to_timedelta(self, values: NDArrayFloat) -> NDArrayTimedelta:
    """
        Convert sequence of numerical values to timedelta
        """
    if self.package == 'pandas':
        return np.array([pd.Timedelta(int(x * self.factor), unit='ns') for x in values])
    else:
        return np.array([timedelta(seconds=x * self.factor) for x in values])