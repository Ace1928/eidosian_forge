import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_subclass_unstack_multi(self):
    df = tm.SubclassedDataFrame([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]], index=MultiIndex.from_tuples(list(zip(list('AABB'), list('cdcd'))), names=['aaa', 'ccc']), columns=MultiIndex.from_tuples(list(zip(list('WWXX'), list('yzyz'))), names=['www', 'yyy']))
    exp = tm.SubclassedDataFrame([[10, 20, 11, 21, 12, 22, 13, 23], [30, 40, 31, 41, 32, 42, 33, 43]], index=Index(['A', 'B'], name='aaa'), columns=MultiIndex.from_tuples(list(zip(list('WWWWXXXX'), list('yyzzyyzz'), list('cdcdcdcd'))), names=['www', 'yyy', 'ccc']))
    res = df.unstack()
    tm.assert_frame_equal(res, exp)
    res = df.unstack('ccc')
    tm.assert_frame_equal(res, exp)
    exp = tm.SubclassedDataFrame([[10, 30, 11, 31, 12, 32, 13, 33], [20, 40, 21, 41, 22, 42, 23, 43]], index=Index(['c', 'd'], name='ccc'), columns=MultiIndex.from_tuples(list(zip(list('WWWWXXXX'), list('yyzzyyzz'), list('ABABABAB'))), names=['www', 'yyy', 'aaa']))
    res = df.unstack('aaa')
    tm.assert_frame_equal(res, exp)