import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('i', [(Timestamp('2011-01-15 12:50:28.502376'), Timestamp('2011-01-20 12:50:28.593448')), (24650000000000001, 24650000000000002)])
def test_groupby_non_arithmetic_agg_int_like_precision(i):
    df = DataFrame([{'a': 1, 'b': i[0]}, {'a': 1, 'b': i[1]}])
    grp_exp = {'first': {'expected': i[0]}, 'last': {'expected': i[1]}, 'min': {'expected': i[0]}, 'max': {'expected': i[1]}, 'nth': {'expected': i[1], 'args': [1]}, 'count': {'expected': 2}}
    for method, data in grp_exp.items():
        if 'args' not in data:
            data['args'] = []
        grouped = df.groupby('a')
        res = getattr(grouped, method)(*data['args'])
        assert res.iloc[0].b == data['expected']