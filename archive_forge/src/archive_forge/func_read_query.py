from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
def read_query(self, sql, index_col=None, coerce_float: bool=True, parse_dates=None, params=None, chunksize: int | None=None, dtype: DtypeArg | None=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> DataFrame | Iterator[DataFrame]:
    cursor = self.execute(sql, params)
    columns = [col_desc[0] for col_desc in cursor.description]
    if chunksize is not None:
        return self._query_iterator(cursor, chunksize, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype, dtype_backend=dtype_backend)
    else:
        data = self._fetchall_as_list(cursor)
        cursor.close()
        frame = _wrap_result(data, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype, dtype_backend=dtype_backend)
        return frame