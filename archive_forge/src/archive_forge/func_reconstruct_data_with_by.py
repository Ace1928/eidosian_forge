from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.missing import remove_na_arraylike
from pandas import (
from pandas.plotting._matplotlib.misc import unpack_single_str_list
def reconstruct_data_with_by(data: DataFrame, by: IndexLabel, cols: IndexLabel) -> DataFrame:
    """
    Internal function to group data, and reassign multiindex column names onto the
    result in order to let grouped data be used in _compute_plot_data method.

    Parameters
    ----------
    data : Original DataFrame to plot
    by : grouped `by` parameter selected by users
    cols : columns of data set (excluding columns used in `by`)

    Returns
    -------
    Output is the reconstructed DataFrame with MultiIndex columns. The first level
    of MI is unique values of groups, and second level of MI is the columns
    selected by users.

    Examples
    --------
    >>> d = {'h': ['h1', 'h1', 'h2'], 'a': [1, 3, 5], 'b': [3, 4, 6]}
    >>> df = pd.DataFrame(d)
    >>> reconstruct_data_with_by(df, by='h', cols=['a', 'b'])
       h1      h2
       a     b     a     b
    0  1.0   3.0   NaN   NaN
    1  3.0   4.0   NaN   NaN
    2  NaN   NaN   5.0   6.0
    """
    by_modified = unpack_single_str_list(by)
    grouped = data.groupby(by_modified)
    data_list = []
    for key, group in grouped:
        columns = MultiIndex.from_product([[key], cols])
        sub_group = group[cols]
        sub_group.columns = columns
        data_list.append(sub_group)
    data = concat(data_list, axis=1)
    return data