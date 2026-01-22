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
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'uint64'])
@pytest.mark.parametrize('method,data', [('first', {'df': [{'a': 1, 'b': 1}, {'a': 2, 'b': 3}]}), ('last', {'df': [{'a': 1, 'b': 2}, {'a': 2, 'b': 4}]}), ('min', {'df': [{'a': 1, 'b': 1}, {'a': 2, 'b': 3}]}), ('max', {'df': [{'a': 1, 'b': 2}, {'a': 2, 'b': 4}]}), ('count', {'df': [{'a': 1, 'b': 2}, {'a': 2, 'b': 2}], 'out_type': 'int64'})])
def test_groupby_non_arithmetic_agg_types(dtype, method, data):
    df = DataFrame([{'a': 1, 'b': 1}, {'a': 1, 'b': 2}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}])
    df['b'] = df.b.astype(dtype)
    if 'args' not in data:
        data['args'] = []
    if 'out_type' in data:
        out_type = data['out_type']
    else:
        out_type = dtype
    exp = data['df']
    df_out = DataFrame(exp)
    df_out['b'] = df_out.b.astype(out_type)
    df_out.set_index('a', inplace=True)
    grpd = df.groupby('a')
    t = getattr(grpd, method)(*data['args'])
    tm.assert_frame_equal(t, df_out)