import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('ops, names', [([np.sqrt], ['sqrt']), ([np.abs, np.sqrt], ['absolute', 'sqrt']), (np.array([np.sqrt]), ['sqrt']), (np.array([np.abs, np.sqrt]), ['absolute', 'sqrt'])])
def test_transform_listlike(axis, float_frame, ops, names):
    other_axis = 1 if axis in {0, 'index'} else 0
    with np.errstate(all='ignore'):
        expected = zip_frames([op(float_frame) for op in ops], axis=other_axis)
    if axis in {0, 'index'}:
        expected.columns = MultiIndex.from_product([float_frame.columns, names])
    else:
        expected.index = MultiIndex.from_product([float_frame.index, names])
    result = float_frame.transform(ops, axis=axis)
    tm.assert_frame_equal(result, expected)