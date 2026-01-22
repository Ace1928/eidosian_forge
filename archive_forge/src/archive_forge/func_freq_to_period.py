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
def freq_to_period(freq: str | offsets.DateOffset) -> int:
    """
    Convert a pandas frequency to a periodicity

    Parameters
    ----------
    freq : str or offset
        Frequency to convert

    Returns
    -------
    int
        Periodicity of freq

    Notes
    -----
    Annual maps to 1, quarterly maps to 4, monthly to 12, weekly to 52.
    """
    if not isinstance(freq, offsets.DateOffset):
        freq = to_offset(freq)
    assert isinstance(freq, offsets.DateOffset)
    freq = freq.rule_code.upper()
    yearly_freqs = ('A-', 'AS-', 'Y-', 'YS-', 'YE-')
    if freq in ('A', 'Y') or freq.startswith(yearly_freqs):
        return 1
    elif freq == 'Q' or freq.startswith(('Q-', 'QS', 'QE')):
        return 4
    elif freq == 'M' or freq.startswith(('M-', 'MS', 'ME')):
        return 12
    elif freq == 'W' or freq.startswith('W-'):
        return 52
    elif freq == 'D':
        return 7
    elif freq == 'B':
        return 5
    elif freq == 'H':
        return 24
    else:
        raise ValueError('freq {} not understood. Please report if you think this is in error.'.format(freq))