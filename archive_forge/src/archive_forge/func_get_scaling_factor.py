from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
def get_scaling_factor(self, units):
    if self.package == 'pandas':
        return NANOSECONDS[units]
    else:
        return SECONDS[units]