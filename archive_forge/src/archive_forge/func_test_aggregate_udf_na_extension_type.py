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
@pytest.mark.xfail(reason='Not implemented;see GH 31256')
def test_aggregate_udf_na_extension_type():

    def aggfunc(x):
        if all(x > 2):
            return 1
        else:
            return pd.NA
    df = DataFrame({'A': pd.array([1, 2, 3])})
    result = df.groupby([1, 1, 2]).agg(aggfunc)
    expected = DataFrame({'A': pd.array([1, pd.NA], dtype='Int64')}, index=[1, 2])
    tm.assert_frame_equal(result, expected)