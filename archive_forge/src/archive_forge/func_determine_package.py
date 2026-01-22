from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
@classmethod
def determine_package(cls, td: Timedelta) -> Literal['pandas', 'cpython']:
    if hasattr(td, 'components'):
        package = 'pandas'
    elif hasattr(td, 'total_seconds'):
        package = 'cpython'
    else:
        msg = f'{td.__class__} format not yet supported.'
        raise ValueError(msg)
    return package