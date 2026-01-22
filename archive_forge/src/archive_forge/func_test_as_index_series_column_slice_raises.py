from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_as_index_series_column_slice_raises(df):
    grouped = df.groupby('A', as_index=False)
    msg = 'Column\\(s\\) C already selected'
    with pytest.raises(IndexError, match=msg):
        grouped['C'].__getitem__('D')