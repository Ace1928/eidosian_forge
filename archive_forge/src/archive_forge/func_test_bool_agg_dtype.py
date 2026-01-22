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
@pytest.mark.parametrize('op', [lambda x: x.sum(), lambda x: x.cumsum(), lambda x: x.transform('sum'), lambda x: x.transform('cumsum'), lambda x: x.agg('sum'), lambda x: x.agg('cumsum')])
def test_bool_agg_dtype(op):
    df = DataFrame({'a': [1, 1], 'b': [False, True]})
    s = df.set_index('a')['b']
    result = op(df.groupby('a'))['b'].dtype
    assert is_integer_dtype(result)
    result = op(s.groupby('a')).dtype
    assert is_integer_dtype(result)