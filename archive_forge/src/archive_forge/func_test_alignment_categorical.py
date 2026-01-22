from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_alignment_categorical(self):
    cat = Categorical(['3z53', '3z53', 'LoJG', 'LoJG', 'LoJG', 'N503'])
    ser1 = Series(2, index=cat)
    ser2 = Series(2, index=cat[:-1])
    result = ser1 * ser2
    exp_index = ['3z53'] * 4 + ['LoJG'] * 9 + ['N503']
    exp_index = pd.CategoricalIndex(exp_index, categories=cat.categories)
    exp_values = [4.0] * 13 + [np.nan]
    expected = Series(exp_values, exp_index)
    tm.assert_series_equal(result, expected)