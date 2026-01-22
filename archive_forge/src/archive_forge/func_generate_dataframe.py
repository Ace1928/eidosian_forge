import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def generate_dataframe(row_size=NROWS, additional_col_values=None, idx_name=None):
    dates = pandas.date_range('2000', freq='h', periods=row_size)
    data = {'col1': np.arange(row_size) * 10, 'col2': [str(x.date()) for x in dates], 'col3': np.arange(row_size) * 10, 'col4': [str(x.time()) for x in dates], 'col5': [get_random_string() for _ in range(row_size)], 'col6': random_state.uniform(low=0.0, high=10000.0, size=row_size)}
    index = None if idx_name is None else pd.RangeIndex(0, row_size, name=idx_name)
    if additional_col_values is not None:
        assert isinstance(additional_col_values, (list, tuple))
        data.update({'col7': random_state.choice(additional_col_values, size=row_size)})
    return pandas.DataFrame(data, index=index)