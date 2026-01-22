import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_constructor_dict_multiindex(self):
    d = {('a', 'a'): {('i', 'i'): 0, ('i', 'j'): 1, ('j', 'i'): 2}, ('b', 'a'): {('i', 'i'): 6, ('i', 'j'): 5, ('j', 'i'): 4}, ('b', 'c'): {('i', 'i'): 7, ('i', 'j'): 8, ('j', 'i'): 9}}
    _d = sorted(d.items())
    df = DataFrame(d)
    expected = DataFrame([x[1] for x in _d], index=MultiIndex.from_tuples([x[0] for x in _d])).T
    expected.index = MultiIndex.from_tuples(expected.index)
    tm.assert_frame_equal(df, expected)
    d['z'] = {'y': 123.0, ('i', 'i'): 111, ('i', 'j'): 111, ('j', 'i'): 111}
    _d.insert(0, ('z', d['z']))
    expected = DataFrame([x[1] for x in _d], index=Index([x[0] for x in _d], tupleize_cols=False)).T
    expected.index = Index(expected.index, tupleize_cols=False)
    df = DataFrame(d)
    df = df.reindex(columns=expected.columns, index=expected.index)
    tm.assert_frame_equal(df, expected)