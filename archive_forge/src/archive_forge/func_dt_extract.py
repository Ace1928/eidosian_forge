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
def dt_extract(self, obj):
    """
        Extract a date or a time unit from a datetime value.

        Parameters
        ----------
        obj : str
            Datetime unit to extract.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
    exprs = self._index_exprs()
    for col in self.columns:
        exprs[col] = build_dt_expr(obj, self.ref(col))
    new_op = TransformNode(self, exprs)
    dtypes = self._dtypes_for_exprs(exprs)
    return self.__constructor__(columns=self.columns, dtypes=dtypes, op=new_op, index=self._index_cache, index_cols=self._index_cols, force_execution_mode=self._force_execution_mode)