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
def join_keys(x, y, by=None):
    """
    Join keys.

    Given two data frames, create a unique key for each row.

    Parameters
    -----------
    x : dataframe
    y : dataframe
    by : list-like
        Column names to join by

    Returns
    -------
    out : dict
        Dictionary with keys x and y. The values of both keys
        are arrays with integer elements. Identical rows in
        x and y dataframes would have the same key in the
        output. The key elements start at 1.
    """
    if by is None:
        by = slice(None, None, None)
    if isinstance(by, tuple):
        by = list(by)
    joint = pd.concat([x[by], y[by]], ignore_index=True)
    keys = ninteraction(joint, drop=True)
    keys = np.asarray(keys)
    nx, ny = (len(x), len(y))
    return {'x': keys[np.arange(nx)], 'y': keys[nx + np.arange(ny)]}