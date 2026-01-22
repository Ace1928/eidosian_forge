import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('reduction', [lambda x: x.mean(), lambda x: x.min(), lambda x: x.max(), lambda x: x.sum()])
def test_numba_vs_python_reductions(reduction, apply_axis):
    df = DataFrame(np.ones((4, 4), dtype=np.float64))
    result = df.apply(reduction, engine='numba', axis=apply_axis)
    expected = df.apply(reduction, engine='python', axis=apply_axis)
    tm.assert_series_equal(result, expected)