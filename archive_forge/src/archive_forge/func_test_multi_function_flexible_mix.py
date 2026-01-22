import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_multi_function_flexible_mix(df):
    grouped = df.groupby('A')
    d = {'C': {'foo': 'mean', 'bar': 'std'}, 'D': {'sum': 'sum'}}
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate(d)
    d = {'C': {'foo': 'mean', 'bar': 'std'}, 'D': 'sum'}
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate(d)
    d = {'C': {'foo': 'mean', 'bar': 'std'}, 'D': 'sum'}
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate(d)