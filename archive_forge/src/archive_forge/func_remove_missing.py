from __future__ import annotations
import inspect
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping import aes
def remove_missing(data: pd.DataFrame, na_rm: bool=False, vars: Sequence[str] | None=None, name: str='', finite: bool=False) -> pd.DataFrame:
    """
    Convenience function to remove missing values from a dataframe

    Parameters
    ----------
    df : dataframe
    na_rm : bool
        If False remove all non-complete rows with and show warning.
    vars : list-like
        columns to act on
    name : str
        Name of calling method for a more informative message
    finite : bool
        If True replace the infinite values in addition to the NaNs
    """
    n = len(data)
    if vars is None:
        vars = data.columns.to_list()
    else:
        vars = data.columns.intersection(list(vars)).to_list()
    if finite:
        lst = [np.inf, -np.inf]
        to_replace = {v: lst for v in vars}
        data.replace(to_replace, np.nan, inplace=True)
        txt = 'non-finite'
    else:
        txt = 'missing'
    data = data.dropna(subset=vars)
    data.reset_index(drop=True, inplace=True)
    if len(data) < n and (not na_rm):
        msg = '{} : Removed {} rows containing {} values.'
        warn(msg.format(name, n - len(data), txt), PlotnineWarning, stacklevel=3)
    return data