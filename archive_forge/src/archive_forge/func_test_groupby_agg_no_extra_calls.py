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
def test_groupby_agg_no_extra_calls():
    df = DataFrame({'key': ['a', 'b', 'c', 'c'], 'value': [1, 2, 3, 4]})
    gb = df.groupby('key')['value']

    def dummy_func(x):
        assert len(x) != 0
        return x.sum()
    gb.agg(dummy_func)