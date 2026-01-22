from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('name_in1,name_in2,name_in3,name_out', [('idx', 'idx', 'idx', 'idx'), ('idx', 'idx', None, None), ('idx', None, None, None), ('idx1', 'idx2', None, None), ('idx1', 'idx1', 'idx2', None), ('idx1', 'idx2', 'idx3', None), (None, None, None, None)])
def test_concat_same_index_names(self, name_in1, name_in2, name_in3, name_out):
    indices = [Index(['a', 'b', 'c'], name=name_in1), Index(['b', 'c', 'd'], name=name_in2), Index(['c', 'd', 'e'], name=name_in3)]
    frames = [DataFrame({c: [0, 1, 2]}, index=i) for i, c in zip(indices, ['x', 'y', 'z'])]
    result = concat(frames, axis=1)
    exp_ind = Index(['a', 'b', 'c', 'd', 'e'], name=name_out)
    expected = DataFrame({'x': [0, 1, 2, np.nan, np.nan], 'y': [np.nan, 0, 1, 2, np.nan], 'z': [np.nan, np.nan, 0, 1, 2]}, index=exp_ind)
    tm.assert_frame_equal(result, expected)