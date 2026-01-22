import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_object_array_eq_ne(self, parser, engine, using_infer_string):
    df = DataFrame({'a': list('aaaabbbbcccc'), 'b': list('aabbccddeeff'), 'c': np.random.default_rng(2).integers(5, size=12), 'd': np.random.default_rng(2).integers(9, size=12)})
    warning = RuntimeWarning if using_infer_string and engine == 'numexpr' else None
    with tm.assert_produces_warning(warning):
        res = df.query('a == b', parser=parser, engine=engine)
    exp = df[df.a == df.b]
    tm.assert_frame_equal(res, exp)
    with tm.assert_produces_warning(warning):
        res = df.query('a != b', parser=parser, engine=engine)
    exp = df[df.a != df.b]
    tm.assert_frame_equal(res, exp)