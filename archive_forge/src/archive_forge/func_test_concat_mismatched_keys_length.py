from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_mismatched_keys_length():
    ser = Series(range(5))
    sers = [ser + n for n in range(4)]
    keys = ['A', 'B', 'C']
    msg = 'The behavior of pd.concat with len\\(keys\\) != len\\(objs\\) is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        concat(sers, keys=keys, axis=1)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        concat(sers, keys=keys, axis=0)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        concat((x for x in sers), keys=(y for y in keys), axis=1)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        concat((x for x in sers), keys=(y for y in keys), axis=0)