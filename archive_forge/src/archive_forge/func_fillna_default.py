from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
def fillna_default(self, col: Any) -> Any:
    """Fill column with default values according to the dtype of the column.

        :param col: series of a pandas like dataframe
        :return: filled series
        """
    dtype = col.dtype
    if pd.api.types.is_datetime64_dtype(dtype):
        return col.fillna(_DEFAULT_DATETIME)
    if pd.api.types.is_string_dtype(dtype):
        return col.fillna('')
    if pd.api.types.is_bool_dtype(dtype):
        return col.fillna(False)
    return col.fillna(0)