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
def build_if_then_else(cond, then_val, else_val, res_type):
    """
    Build a conditional operator expression.

    Parameters
    ----------
    cond : BaseExpr
        A condition to check.
    then_val : BaseExpr
        A value to use for passed condition.
    else_val : BaseExpr
        A value to use for failed condition.
    res_type : dtype
        The result data type.

    Returns
    -------
    BaseExpr
        The conditional operator expression.
    """
    if is_datetime64_dtype(res_type):
        if then_val._dtype != res_type:
            then_val = then_val.cast(res_type)
        if else_val._dtype != res_type:
            else_val = else_val.cast(res_type)
    return OpExpr('CASE', [cond, then_val, else_val], res_type)