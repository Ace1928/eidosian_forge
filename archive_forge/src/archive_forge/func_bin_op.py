import re
from typing import Hashable, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow
from pandas._libs.lib import no_default
from pandas.core.dtypes.common import (
from pandas.core.indexes.api import Index, MultiIndex, RangeIndex
from pyarrow.types import is_dictionary
from modin.core.dataframe.base.dataframe.utils import (
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.dataframe.pandas.metadata.dtypes import get_categories_dtype
from modin.core.dataframe.pandas.utils import concatenate
from modin.error_message import ErrorMessage
from modin.experimental.core.storage_formats.hdk.query_compiler import (
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
from ..db_worker import DbTable
from ..df_algebra import (
from ..expr import (
from ..partitioning.partition_manager import HdkOnNativeDataframePartitionManager
from .utils import (
def bin_op(self, other, op_name, **kwargs):
    """
        Perform binary operation.

        An arithmetic binary operation or a comparison operation to
        perform on columns.

        Parameters
        ----------
        other : scalar, list-like, or HdkOnNativeDataframe
            The second operand.
        op_name : str
            An operation to perform.
        **kwargs : dict
            Keyword args.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
    if isinstance(other, (int, float, str)):
        value_expr = LiteralExpr(other)
        exprs = self._index_exprs()
        for col in self.columns:
            exprs[col] = self.ref(col).bin_op(value_expr, op_name)
        return self.__constructor__(columns=self.columns, dtypes=self._dtypes_for_exprs(exprs), op=TransformNode(self, exprs), index=self._index_cache, index_cols=self._index_cols, force_execution_mode=self._force_execution_mode)
    elif isinstance(other, list):
        if kwargs.get('axis', 1) == 0:
            raise NotImplementedError(f'{op_name} on rows')
        if len(other) != len(self.columns):
            raise ValueError(f'length must be {len(self.columns)}: given {len(other)}')
        exprs = self._index_exprs()
        for col, val in zip(self.columns, other):
            exprs[col] = self.ref(col).bin_op(LiteralExpr(val), op_name)
        return self.__constructor__(columns=self.columns, dtypes=self._dtypes_for_exprs(exprs), op=TransformNode(self, exprs), index=self._index_cache, index_cols=self._index_cols, force_execution_mode=self._force_execution_mode)
    elif isinstance(other, type(self)):
        base = self._find_common_projections_base(other)
        if base is None:
            raise NotImplementedError('unsupported binary op args (outer join is not supported)')
        new_columns = self.columns.tolist()
        for col in other.columns:
            if col not in self.columns:
                new_columns.append(col)
        new_columns = sorted(new_columns)
        fill_value = kwargs.get('fill_value', None)
        if fill_value is not None:
            fill_value = LiteralExpr(fill_value)
        if is_cmp_op(op_name):
            null_value = LiteralExpr(op_name == 'ne')
        else:
            null_value = LiteralExpr(None)
        exprs = self._index_exprs()
        for col in new_columns:
            lhs = self.ref(col) if col in self.columns else fill_value
            rhs = other.ref(col) if col in other.columns else fill_value
            if lhs is None or rhs is None:
                exprs[col] = null_value
            else:
                exprs[col] = lhs.bin_op(rhs, op_name)
        exprs = translate_exprs_to_base(exprs, base)
        return self.__constructor__(columns=new_columns, dtypes=self._dtypes_for_exprs(exprs), op=TransformNode(base, exprs), index=self._index_cache, index_cols=self._index_cols, force_execution_mode=self._force_execution_mode)
    else:
        raise NotImplementedError(f'unsupported operand type: {type(other)}')