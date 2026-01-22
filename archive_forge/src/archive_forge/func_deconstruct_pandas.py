from __future__ import annotations
import datetime as dt
import sys
from enum import Enum
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from pyviz_comms import JupyterComm
from ..io.state import state
from ..reactive import ReactiveData
from ..util import datetime_types, lazy_load
from ..viewable import Viewable
from .base import ModelPane
def deconstruct_pandas(data, kwargs=None):
    """
    Given a dataframe, flatten it by resetting the index and memoizing
    the pivots that were applied.

    This code was copied from the Perspective repository and is
    reproduced under Apache 2.0 license. See the original at:

    https://github.com/finos/perspective/blob/master/python/perspective/perspective/core/data/pd.py

    Arguments
    ---------
    data: (pandas.dataframe)
      A Pandas DataFrame to parse

    Returns
    -------
    data: pandas.DataFrame
      A flattened version of the DataFrame
    kwargs: dict
      A dictionary containing optional members `columns`,
      `group_by`, and `split_by`.
    """
    import pandas as pd
    kwargs = kwargs or {}
    kwargs = {'columns': [], 'group_by': [], 'split_by': []}
    if isinstance(data.index, pd.PeriodIndex):
        data.index = data.index.to_timestamp()
    if isinstance(data, pd.DataFrame):
        if hasattr(pd, 'CategoricalDtype'):
            for k, v in data.dtypes.items():
                if isinstance(v, pd.CategoricalDtype):
                    data[k] = data[k].astype(str)
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex) and isinstance(data.index, pd.MultiIndex):
        kwargs['group_by'].extend([str(c) for c in data.index.names])
        if None in data.columns.names:
            existent = kwargs['group_by'] + data.columns.names
            for c in data.columns.names:
                if c is not None:
                    kwargs['split_by'].append(c)
                    data = data.stack()
            data = pd.DataFrame(data).reset_index()
            for new_column in data.columns:
                if new_column not in existent:
                    kwargs['columns'].append(new_column)
        else:
            for _ in kwargs['group_by']:
                data = data.unstack()
            data = pd.DataFrame(data)
        i = 0
        new_names = list(data.index.names)
        for j, val in enumerate(data.index.names):
            if val is None:
                new_names[j] = 'index' if i == 0 else 'index-{}'.format(i)
                i += 1
            elif str(val) not in kwargs['group_by']:
                kwargs['split_by'].append(str(val))
        data.index.names = new_names
        data = data.reset_index()
        data.columns = [str(c) if c in ['index'] + kwargs['group_by'] + kwargs['split_by'] + kwargs['columns'] else 'value' for c in data.columns]
        kwargs['columns'].extend(['value' for c in data.columns if c not in ['index'] + kwargs['group_by'] + kwargs['split_by'] + kwargs['columns']])
    elif isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        if data.index.name:
            kwargs['group_by'].append(str(data.index.name))
            push_row_pivot = False
        else:
            push_row_pivot = True
        data = pd.DataFrame(data.unstack())
        i = 0
        new_names = list(data.index.names)
        for j, val in enumerate(data.index.names):
            if val is None:
                new_names[j] = 'index' if i == 0 else 'index-{}'.format(i)
                i += 1
                if push_row_pivot:
                    kwargs['group_by'].append(str(new_names[j]))
            elif str(val) not in kwargs['group_by']:
                kwargs['split_by'].append(str(val))
        data.index.names = new_names
        data.columns = [str(c) if c in ['index'] + kwargs['group_by'] + kwargs['split_by'] else 'value' for c in data.columns]
        kwargs['columns'].extend(['value' for c in data.columns if c not in ['index'] + kwargs['group_by'] + kwargs['split_by']])
    elif isinstance(data, pd.DataFrame) and isinstance(data.index, pd.MultiIndex):
        kwargs['group_by'].extend(list(data.index.names))
        data = data.reset_index()
    if isinstance(data, pd.DataFrame):
        if 'index' not in [str(c).lower() for c in data.columns]:
            data = data.reset_index(col_fill='index')
        if not kwargs['columns']:
            kwargs['columns'].extend([str(c) for c in data.columns])
            data.columns = kwargs['columns']
    if isinstance(data, pd.Series):
        flattened = data.reset_index()
        if isinstance(data, pd.Series):
            flattened.name = data.name
            flattened.columns = [str(c) for c in flattened.columns]
        data = flattened
    return (data, kwargs)