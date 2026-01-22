from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
def scaled_limits(self) -> TupleFloat2:
    """
        Minimum and Maximum to use for computing breaks
        """
    _min = self.limits[0] / self.factor
    _max = self.limits[1] / self.factor
    return (_min, _max)