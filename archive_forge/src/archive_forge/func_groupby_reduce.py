import datetime
import re
from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.api.types import is_object_dtype
from pandas.core.dtypes.common import is_dtype_equal, is_list_like, is_numeric_dtype
from pandas.core.indexes.api import Index, RangeIndex
from modin.config import Engine, IsRayCluster, MinPartitionSize, NPartitions
from modin.core.dataframe.base.dataframe.dataframe import ModinDataframe
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType, is_trivial_index
from modin.core.dataframe.pandas.dataframe.utils import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.storage_formats.pandas.utils import get_length_list
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none, is_full_grab_slice
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@lazy_metadata_decorator(apply_axis='both')
def groupby_reduce(self, axis, by, map_func, reduce_func, new_index=None, new_columns=None, apply_indices=None):
    """
        Groupby another Modin DataFrame dataframe and aggregate the result.

        Parameters
        ----------
        axis : {0, 1}
            Axis to groupby and aggregate over.
        by : PandasDataframe or None
            A Modin DataFrame to group by.
        map_func : callable
            Map component of the aggregation.
        reduce_func : callable
            Reduce component of the aggregation.
        new_index : pandas.Index, optional
            Index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : pandas.Index, optional
            Columns of the result. We may know this in advance,
            and if not provided it must be computed.
        apply_indices : list-like, default: None
            Indices of `axis ^ 1` to apply groupby over.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
    by_parts = by if by is None else by._partitions
    if by is None:
        self._propagate_index_objs(axis=0)
    if apply_indices is not None:
        numeric_indices = self.get_axis(axis ^ 1).get_indexer_for(apply_indices)
        apply_indices = list(self._get_dict_of_block_index(axis ^ 1, numeric_indices).keys())
    if by_parts is not None:
        if by_parts.shape[axis] != self._partitions.shape[axis]:
            self._filter_empties(compute_metadata=False)
    new_partitions = self._partition_mgr_cls.groupby_reduce(axis, self._partitions, by_parts, map_func, reduce_func, apply_indices)
    return self.__constructor__(new_partitions, index=new_index, columns=new_columns)