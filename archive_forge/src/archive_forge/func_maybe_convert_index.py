from __future__ import annotations
import functools
from typing import (
import warnings
import numpy as np
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.converter import (
from pandas.tseries.frequencies import (
def maybe_convert_index(ax: Axes, data: NDFrameT) -> NDFrameT:
    if isinstance(data.index, (ABCDatetimeIndex, ABCPeriodIndex)):
        freq: str | BaseOffset | None = data.index.freq
        if freq is None:
            data.index = cast('DatetimeIndex', data.index)
            freq = data.index.inferred_freq
            freq = to_offset(freq)
        if freq is None:
            freq = _get_ax_freq(ax)
        if freq is None:
            raise ValueError('Could not get frequency alias for plotting')
        freq_str = _get_period_alias(freq)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'PeriodDtype\\[B\\] is deprecated', category=FutureWarning)
            if isinstance(data.index, ABCDatetimeIndex):
                data = data.tz_localize(None).to_period(freq=freq_str)
            elif isinstance(data.index, ABCPeriodIndex):
                data.index = data.index.asfreq(freq=freq_str)
    return data