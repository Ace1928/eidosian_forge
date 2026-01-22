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
def test_groupby_series_with_name(df):
    result = df.groupby(df['A']).mean(numeric_only=True)
    result2 = df.groupby(df['A'], as_index=False).mean(numeric_only=True)
    assert result.index.name == 'A'
    assert 'A' in result2
    result = df.groupby([df['A'], df['B']]).mean()
    result2 = df.groupby([df['A'], df['B']], as_index=False).mean()
    assert result.index.names == ('A', 'B')
    assert 'A' in result2
    assert 'B' in result2