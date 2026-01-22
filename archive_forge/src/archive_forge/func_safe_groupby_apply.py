from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
def safe_groupby_apply(self, df: T, cols: List[str], func: Callable[[T], T], key_col_name='__safe_groupby_key__', **kwargs: Any) -> T:
    """Safe groupby apply operation on pandas like dataframes.
        In pandas like groupby apply, if any key is null, the whole group is dropped.
        This method makes sure those groups are included.

        :param df: pandas like dataframe
        :param cols: columns to group on, can be empty
        :param func: apply function, df in, df out
        :param key_col_name: temp key as index for groupu.
            default "__safe_groupby_key__"
        :return: output dataframe

        :Notice:
        The dataframe must be either empty, or with type pd.RangeIndex, pd.Int64Index
        or pd.UInt64Index and without a name, otherwise, `ValueError` will raise.
        """

    def _wrapper(df: T) -> T:
        return func(df.reset_index(drop=True))
    self.ensure_compatible(df)
    if len(cols) == 0:
        return func(df)
    return df.groupby(cols, dropna=False, group_keys=False).apply(lambda df: _wrapper(df), **kwargs).reset_index(drop=True)