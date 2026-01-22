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
def copy_missing_columns(df, ref_df):
    """
    Copy missing columns from ref_df to df

    If df and ref_df are the same length, the columns are
    copied in the entirety. If the length ofref_df is a
    divisor of the length of df, then the values of the
    columns from ref_df are repeated.

    Otherwise if not the same length, df gets a column
    where all elements are the same as the first element
    in ref_df

    Parameters
    ----------
    df : dataframe
        Dataframe to which columns will be added
    ref_df : dataframe
        Dataframe from which columns will be copied
    """
    cols = ref_df.columns.difference(df.columns)
    _loc = ref_df.columns.get_loc
    l1, l2 = (len(df), len(ref_df))
    if l1 >= l2 and l1 % l2 == 0:
        idx = np.tile(range(l2), l1 // l2)
    else:
        idx = np.repeat(0, l1)
    for col in cols:
        df[col] = ref_df.iloc[idx, _loc(col)].to_numpy()