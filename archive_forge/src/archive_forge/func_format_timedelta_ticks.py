from __future__ import annotations
import contextlib
import datetime as pydt
from datetime import (
import functools
from typing import (
import warnings
import matplotlib.dates as mdates
from matplotlib.ticker import (
from matplotlib.transforms import nonsingular
import matplotlib.units as munits
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._typing import (
from pandas.core.dtypes.common import (
from pandas import (
import pandas.core.common as com
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
import pandas.core.tools.datetimes as tools
@staticmethod
def format_timedelta_ticks(x, pos, n_decimals: int) -> str:
    """
        Convert seconds to 'D days HH:MM:SS.F'
        """
    s, ns = divmod(x, 10 ** 9)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    decimals = int(ns * 10 ** (n_decimals - 9))
    s = f'{int(h):02d}:{int(m):02d}:{int(s):02d}'
    if n_decimals > 0:
        s += f'.{decimals:0{n_decimals}d}'
    if d != 0:
        s = f'{int(d):d} days {s}'
    return s