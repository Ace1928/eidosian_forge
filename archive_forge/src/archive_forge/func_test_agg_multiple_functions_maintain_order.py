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
def test_agg_multiple_functions_maintain_order(df):
    funcs = [('mean', np.mean), ('max', np.max), ('min', np.min)]
    msg = 'is currently using SeriesGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby('A')['C'].agg(funcs)
    exp_cols = Index(['mean', 'max', 'min'])
    tm.assert_index_equal(result.columns, exp_cols)