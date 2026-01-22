import ast
import hashlib
import re
import warnings
from collections.abc import Iterable
from typing import Hashable, List
import numpy as np
import pandas
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.groupby.base import transformation_kernels
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.core.indexing import check_bool_indexer
from pandas.errors import DataError
from modin.config import CpuCount, RangePartitioning, use_range_partitioning_groupby
from modin.core.dataframe.algebra import (
from modin.core.dataframe.algebra.default2pandas.groupby import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import get_logger
from modin.utils import (
from .aggregations import CorrCovBuilder
from .groupby import GroupbyReduceImpl, PivotTableImpl
from .merge import MergeImpl
from .utils import get_group_names, merge_partitioning
def is_tree_like_or_1d(calc_index, valid_index):
    """
            Check whether specified index is a single dimensional or built in a tree manner.

            Parameters
            ----------
            calc_index : pandas.Index
                Frame index to check.
            valid_index : pandas.Index
                Frame index on the opposite from `calc_index` axis.

            Returns
            -------
            bool
                True if `calc_index` is not MultiIndex or MultiIndex and built in a tree manner.
                False otherwise.
            """
    if not isinstance(calc_index, pandas.MultiIndex):
        return True
    actual_len = 1
    for lvl in calc_index.levels:
        actual_len *= len(lvl)
    return len(self.index) * len(self.columns) == actual_len * len(valid_index)