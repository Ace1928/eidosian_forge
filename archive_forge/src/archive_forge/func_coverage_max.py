from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
def coverage_max(self, dmin: float, dmax: float, span: float) -> float:
    range = dmax - dmin
    if span > range:
        half = (span - range) / 2.0
        return 1 - half ** 2 / (0.1 * range) ** 2
    else:
        return 1