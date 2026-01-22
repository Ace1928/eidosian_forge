from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_nanvar_nans(self, samples, variance):
    samples_test = np.nan * np.ones(2 * samples.shape[0])
    samples_test[::2] = samples
    actual_variance = nanops.nanvar(samples_test, skipna=True)
    tm.assert_almost_equal(actual_variance, variance, rtol=0.01)
    actual_variance = nanops.nanvar(samples_test, skipna=False)
    tm.assert_almost_equal(actual_variance, np.nan, rtol=0.01)