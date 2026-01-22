from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
def combine_dtypes(row):
    if (row == 'unknown').any():
        return 'unknown'
    row = row.fillna(pandas.api.types.pandas_dtype('float'))
    return find_common_type(list(row.values))