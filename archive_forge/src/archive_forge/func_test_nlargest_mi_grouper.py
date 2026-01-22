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
def test_nlargest_mi_grouper():
    npr = np.random.RandomState(123456789)
    dts = date_range('20180101', periods=10)
    iterables = [dts, ['one', 'two']]
    idx = MultiIndex.from_product(iterables, names=['first', 'second'])
    s = Series(npr.randn(20), index=idx)
    result = s.groupby('first').nlargest(1)
    exp_idx = MultiIndex.from_tuples([(dts[0], dts[0], 'one'), (dts[1], dts[1], 'one'), (dts[2], dts[2], 'one'), (dts[3], dts[3], 'two'), (dts[4], dts[4], 'one'), (dts[5], dts[5], 'one'), (dts[6], dts[6], 'one'), (dts[7], dts[7], 'one'), (dts[8], dts[8], 'two'), (dts[9], dts[9], 'one')], names=['first', 'first', 'second'])
    exp_values = [2.2129019979039612, 1.8417114045748335, 0.858963679564603, 1.3759151378258088, 0.9430284594687134, 0.5296914208183142, 0.8318045593815487, -0.8476703342910327, 0.3804446884133735, -0.8028845810770998]
    expected = Series(exp_values, index=exp_idx)
    tm.assert_series_equal(result, expected, check_exact=False, rtol=0.001)