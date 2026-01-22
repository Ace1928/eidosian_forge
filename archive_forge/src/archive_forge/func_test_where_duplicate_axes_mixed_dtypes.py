from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_duplicate_axes_mixed_dtypes():
    result = DataFrame(data=[[0, np.nan]], columns=Index(['A', 'A']))
    index, columns = result.axes
    mask = DataFrame(data=[[True, True]], columns=columns, index=index)
    a = result.astype(object).where(mask)
    b = result.astype('f8').where(mask)
    c = result.T.where(mask.T).T
    d = result.where(mask)
    tm.assert_frame_equal(a.astype('f8'), b.astype('f8'))
    tm.assert_frame_equal(b.astype('f8'), c.astype('f8'))
    tm.assert_frame_equal(c.astype('f8'), d.astype('f8'))