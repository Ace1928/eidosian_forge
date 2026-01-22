from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_agg_multiple_mixed_raises():
    mdf = DataFrame({'A': [1, 2, 3], 'B': [1.0, 2.0, 3.0], 'C': ['foo', 'bar', 'baz'], 'D': date_range('20130101', periods=3)})
    msg = 'does not support reduction'
    with pytest.raises(TypeError, match=msg):
        mdf.agg(['min', 'sum'])
    with pytest.raises(TypeError, match=msg):
        mdf[['D', 'C', 'B', 'A']].agg(['sum', 'min'])