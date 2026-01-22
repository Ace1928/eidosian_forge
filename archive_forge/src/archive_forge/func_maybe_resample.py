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
def maybe_resample(series: Series, ax: Axes, kwargs: dict[str, Any]):
    if 'how' in kwargs:
        raise ValueError("'how' is not a valid keyword for plotting functions. If plotting multiple objects on shared axes, resample manually first.")
    freq, ax_freq = _get_freq(ax, series)
    if freq is None:
        raise ValueError('Cannot use dynamic axis without frequency info')
    if isinstance(series.index, ABCDatetimeIndex):
        series = series.to_period(freq=freq)
    if ax_freq is not None and freq != ax_freq:
        if is_superperiod(freq, ax_freq):
            series = series.copy()
            series.index = series.index.asfreq(ax_freq, how='s')
            freq = ax_freq
        elif _is_sup(freq, ax_freq):
            ser_ts = series.to_timestamp()
            ser_d = ser_ts.resample('D').last().dropna()
            ser_freq = ser_d.resample(ax_freq).last().dropna()
            series = ser_freq.to_period(ax_freq)
            freq = ax_freq
        elif is_subperiod(freq, ax_freq) or _is_sub(freq, ax_freq):
            _upsample_others(ax, freq, kwargs)
        else:
            raise ValueError('Incompatible frequency conversion')
    return (freq, series)