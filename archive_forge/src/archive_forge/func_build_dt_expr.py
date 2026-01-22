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
def build_dt_expr(dt_operation, col_expr):
    """
    Build a datetime extraction expression.

    Parameters
    ----------
    dt_operation : str
        Datetime field to extract.
    col_expr : BaseExpr
        An expression to extract from.

    Returns
    -------
    BaseExpr
        The extract expression.
    """
    operation = LiteralExpr(dt_operation)
    res = OpExpr('PG_EXTRACT', [operation, col_expr], _get_dtype('int32'))
    if dt_operation == 'isodow':
        res = res.sub(LiteralExpr(1))
    elif dt_operation == 'microsecond':
        res = res.mod(LiteralExpr(1000000))
    elif dt_operation == 'nanosecond':
        res = res.mod(LiteralExpr(1000))
    return res