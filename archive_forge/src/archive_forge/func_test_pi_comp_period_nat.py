import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
def test_pi_comp_period_nat(self):
    idx = PeriodIndex(['2011-01', 'NaT', '2011-03', '2011-04'], freq='M', name='idx')
    per = idx[2]
    f = lambda x: x == per
    exp = np.array([False, False, True, False], dtype=np.bool_)
    self._check(idx, f, exp)
    f = lambda x: per == x
    self._check(idx, f, exp)
    f = lambda x: x == pd.NaT
    exp = np.array([False, False, False, False], dtype=np.bool_)
    self._check(idx, f, exp)
    f = lambda x: pd.NaT == x
    self._check(idx, f, exp)
    f = lambda x: x != per
    exp = np.array([True, True, False, True], dtype=np.bool_)
    self._check(idx, f, exp)
    f = lambda x: per != x
    self._check(idx, f, exp)
    f = lambda x: x != pd.NaT
    exp = np.array([True, True, True, True], dtype=np.bool_)
    self._check(idx, f, exp)
    f = lambda x: pd.NaT != x
    self._check(idx, f, exp)
    f = lambda x: per >= x
    exp = np.array([True, False, True, False], dtype=np.bool_)
    self._check(idx, f, exp)
    f = lambda x: x < per
    exp = np.array([True, False, False, False], dtype=np.bool_)
    self._check(idx, f, exp)
    f = lambda x: x > pd.NaT
    exp = np.array([False, False, False, False], dtype=np.bool_)
    self._check(idx, f, exp)
    f = lambda x: pd.NaT >= x
    exp = np.array([False, False, False, False], dtype=np.bool_)
    self._check(idx, f, exp)