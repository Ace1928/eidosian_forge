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
def sort_rows(self, columns, ascending, ignore_index, na_position):
    """
        Sort rows of the frame.

        Parameters
        ----------
        columns : str or list of str
            Sorting keys.
        ascending : bool or list of bool
            Sort order.
        ignore_index : bool
            Drop index columns.
        na_position : {"first", "last"}
            NULLs position.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
    if na_position != 'first' and na_position != 'last':
        raise ValueError(f"Unsupported na_position value '{na_position}'")
    base = self
    if not ignore_index and base._index_cols is None:
        base = base._materialize_rowid()
    if not isinstance(columns, list):
        columns = [columns]
    columns = [base._find_index_or_col(col) for col in columns]
    if isinstance(ascending, list):
        if len(ascending) != len(columns):
            raise ValueError("ascending list length doesn't match columns list")
    else:
        if not isinstance(ascending, bool):
            raise ValueError('unsupported ascending value')
        ascending = [ascending] * len(columns)
    if ignore_index:
        if base._index_cols is not None:
            drop_index_cols_before = [col for col in base._index_cols if col not in columns]
            drop_index_cols_after = [col for col in base._index_cols if col in columns]
            if not drop_index_cols_after:
                drop_index_cols_after = None
            if drop_index_cols_before:
                exprs = dict()
                index_cols = drop_index_cols_after if drop_index_cols_after else None
                for col in drop_index_cols_after:
                    exprs[col] = base.ref(col)
                for col in base.columns:
                    exprs[col] = base.ref(col)
                base = base.__constructor__(columns=base.columns, dtypes=base._dtypes_for_exprs(exprs), op=TransformNode(base, exprs), index_cols=index_cols, force_execution_mode=base._force_execution_mode)
            base = base.__constructor__(columns=base.columns, dtypes=base.copy_dtypes_cache(), op=SortNode(base, columns, ascending, na_position), index_cols=base._index_cols, force_execution_mode=base._force_execution_mode)
            if drop_index_cols_after:
                exprs = dict()
                for col in base.columns:
                    exprs[col] = base.ref(col)
                base = base.__constructor__(columns=base.columns, dtypes=base._dtypes_for_exprs(exprs), op=TransformNode(base, exprs), index_cols=None, force_execution_mode=base._force_execution_mode)
            return base
        else:
            return base.__constructor__(columns=base.columns, dtypes=base.copy_dtypes_cache(), op=SortNode(base, columns, ascending, na_position), index_cols=None, force_execution_mode=base._force_execution_mode)
    else:
        return base.__constructor__(columns=base.columns, dtypes=base.copy_dtypes_cache(), op=SortNode(base, columns, ascending, na_position), index_cols=base._index_cols, force_execution_mode=base._force_execution_mode)