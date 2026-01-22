import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_subclass_stack_multi_mixed(self):
    df = tm.SubclassedDataFrame([[10, 11, 12.0, 13.0], [20, 21, 22.0, 23.0], [30, 31, 32.0, 33.0], [40, 41, 42.0, 43.0]], index=MultiIndex.from_tuples(list(zip(list('AABB'), list('cdcd'))), names=['aaa', 'ccc']), columns=MultiIndex.from_tuples(list(zip(list('WWXX'), list('yzyz'))), names=['www', 'yyy']))
    exp = tm.SubclassedDataFrame([[10, 12.0], [11, 13.0], [20, 22.0], [21, 23.0], [30, 32.0], [31, 33.0], [40, 42.0], [41, 43.0]], index=MultiIndex.from_tuples(list(zip(list('AAAABBBB'), list('ccddccdd'), list('yzyzyzyz'))), names=['aaa', 'ccc', 'yyy']), columns=Index(['W', 'X'], name='www'))
    res = df.stack(future_stack=True)
    tm.assert_frame_equal(res, exp)
    res = df.stack('yyy', future_stack=True)
    tm.assert_frame_equal(res, exp)
    exp = tm.SubclassedDataFrame([[10.0, 11.0], [12.0, 13.0], [20.0, 21.0], [22.0, 23.0], [30.0, 31.0], [32.0, 33.0], [40.0, 41.0], [42.0, 43.0]], index=MultiIndex.from_tuples(list(zip(list('AAAABBBB'), list('ccddccdd'), list('WXWXWXWX'))), names=['aaa', 'ccc', 'www']), columns=Index(['y', 'z'], name='yyy'))
    res = df.stack('www', future_stack=True)
    tm.assert_frame_equal(res, exp)