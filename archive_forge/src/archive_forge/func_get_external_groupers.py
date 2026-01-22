import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
def get_external_groupers(df, columns, drop_from_original_df=False, add_plus_one=False):
    """
    Construct ``by`` argument containing external groupers.

    Parameters
    ----------
    df : pandas.DataFrame or modin.pandas.DataFrame
    columns : list[tuple[bool, str]]
        Columns to group on. If ``True`` do ``df[col]``, otherwise keep the column name.
        '''
        >>> columns = [(True, "a"), (False, "b")]
        >>> get_external_groupers(df, columns)
        [
            pandas.Series(..., name="a"),
            "b"
        ]
        '''
    drop_from_original_df : bool, default: False
        Whether to drop selected external columns from `df`.
    add_plus_one : bool, default: False
        Whether to do ``df[name] + 1`` for external groupers (so they won't be considered as
        sibling with `df`).

    Returns
    -------
    new_df : pandas.DataFrame or modin.pandas.DataFrame
        If `drop_from_original_df` was True, returns a new dataframe with
        dropped external columns, otherwise returns `df`.
    by : list
        Groupers to pass to `df.groupby(by)`.
    """
    new_df = df
    by = []
    for lookup, name in columns:
        if lookup:
            ser = df[name].copy()
            if add_plus_one:
                ser = ser + 1
            by.append(ser)
            if drop_from_original_df:
                new_df = new_df.drop(columns=[name])
        else:
            by.append(name)
    return (new_df, by)