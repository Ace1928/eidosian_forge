import abc
import warnings
from typing import Hashable, List, Optional
import numpy as np
import pandas
import pandas.core.resample
from pandas._typing import DtypeBackend, IndexLabel, Suffixes
from pandas.core.dtypes.common import is_number, is_scalar
from modin.config import StorageFormat
from modin.core.dataframe.algebra.default2pandas import (
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
from . import doc_utils
def insert_item(self, axis, loc, value, how='inner', replace=False):
    """
        Insert rows/columns defined by `value` at the specified position.

        If frames are not aligned along specified axis, perform frames alignment first.

        Parameters
        ----------
        axis : {0, 1}
            Axis to insert along. 0 means insert rows, when 1 means insert columns.
        loc : int
            Position to insert `value`.
        value : BaseQueryCompiler
            Rows/columns to insert.
        how : {"inner", "outer", "left", "right"}, default: "inner"
            Type of join that will be used if frames are not aligned.
        replace : bool, default: False
            Whether to insert item after column/row at `loc-th` position or to replace
            it by `value`.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with inserted values.
        """
    assert isinstance(value, type(self))

    def mask(idx):
        if len(idx) == len(self.get_axis(axis)):
            return self
        return self.getitem_column_array(idx, numeric=True) if axis else self.getitem_row_array(idx)
    if 0 <= loc < len(self.get_axis(axis)):
        first_mask = mask(list(range(loc)))
        second_mask_loc = loc + 1 if replace else loc
        second_mask = mask(list(range(second_mask_loc, len(self.get_axis(axis)))))
        return first_mask.concat(axis, [value, second_mask], join=how, sort=False)
    else:
        return self.concat(axis, [value], join=how, sort=False)