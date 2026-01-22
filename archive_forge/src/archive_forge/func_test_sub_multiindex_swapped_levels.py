from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
def test_sub_multiindex_swapped_levels():
    df = pd.DataFrame({'a': np.random.default_rng(2).standard_normal(6)}, index=pd.MultiIndex.from_product([['a', 'b'], [0, 1, 2]], names=['levA', 'levB']))
    df2 = df.copy()
    df2.index = df2.index.swaplevel(0, 1)
    result = df - df2
    expected = pd.DataFrame([0.0] * 6, columns=['a'], index=df.index)
    tm.assert_frame_equal(result, expected)