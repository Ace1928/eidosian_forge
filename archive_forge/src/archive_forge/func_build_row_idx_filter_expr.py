import abc
from typing import Generator, Type, Union
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.dtypes.common import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import ColNameCodec, to_arrow_type
def build_row_idx_filter_expr(row_idx, row_col):
    """
    Build an expression to filter rows by rowid.

    Parameters
    ----------
    row_idx : int or list of int
        The row numeric indices to select.
    row_col : InputRefExpr
        The rowid column reference expression.

    Returns
    -------
    BaseExpr
        The resulting filtering expression.
    """
    if not is_list_like(row_idx):
        return row_col.eq(row_idx)
    if is_range_like(row_idx):
        start = row_idx[0]
        stop = row_idx[-1]
        step = row_idx.step
        if step < 0:
            start, stop = (stop, start)
            step = -step
        exprs = [row_col.ge(start), row_col.le(stop)]
        if step > 1:
            mod = OpExpr('MOD', [row_col, LiteralExpr(step)], _get_dtype(int))
            exprs.append(mod.eq(0))
        return OpExpr('AND', exprs, _get_dtype(bool))
    exprs = [row_col.eq(idx) for idx in row_idx]
    return OpExpr('OR', exprs, _get_dtype(bool))