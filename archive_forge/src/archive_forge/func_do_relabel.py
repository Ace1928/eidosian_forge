from __future__ import annotations
import warnings
from collections.abc import Iterable
from types import BuiltinFunctionType
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas
import pandas.core.common as com
import pandas.core.groupby
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.dtypes.common import (
from pandas.errors import SpecificationError
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, disable_logging
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import (
from .series import Series
from .utils import is_label
from .window import RollingGroupby
def do_relabel(obj_to_relabel):
    result_labels = [None] * len(old_kwargs)
    for idx, labels in enumerate(old_kwargs.values()):
        if is_scalar(labels) or callable(labels):
            result_labels[idx] = labels if not callable(labels) else labels.__name__
            continue
        new_elem = []
        for label in labels:
            if is_scalar(label) or callable(label):
                new_elem.append(label if not callable(label) else label.__name__)
            else:
                new_elem.extend(label)
        result_labels[idx] = tuple(new_elem)
    new_order = obj_to_relabel.columns.get_indexer(result_labels)
    new_columns_idx = pandas.Index(new_columns)
    if not self._as_index:
        nby_cols = len(obj_to_relabel.columns) - len(new_columns_idx)
        new_order = np.concatenate([np.arange(nby_cols), new_order])
        by_cols = obj_to_relabel.columns[:nby_cols]
        if by_cols.nlevels != new_columns_idx.nlevels:
            by_cols = by_cols.remove_unused_levels()
            empty_levels = [i for i, level in enumerate(by_cols.levels) if len(level) == 1 and level[0] == '']
            by_cols = by_cols.droplevel(empty_levels)
        new_columns_idx = by_cols.append(new_columns_idx)
    result = obj_to_relabel.iloc[:, new_order]
    result.columns = new_columns_idx
    return result