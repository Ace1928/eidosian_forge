import datetime
import pandas
import pytest
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_object_dtype
import modin.pandas as pd
from modin.tests.pandas.utils import df_equals
from modin.tests.pandas.utils import eval_io as general_eval_io
from modin.utils import try_cast_to_pandas
def convert_to_time(value):
    """Convert passed value to `datetime.time`."""
    if isinstance(value, datetime.time):
        return value
    elif isinstance(value, str):
        return datetime.time.fromisoformat(value)
    else:
        return datetime.time(value)