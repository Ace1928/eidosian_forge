import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
@pytest.mark.parametrize('columns', [Index([0, 1], dtype=np.int64), Index([0.0, 1.0], dtype=np.float64), date_range('2020-01-01', periods=2), timedelta_range('1 day', periods=2), period_range('2020-01-01', periods=2, freq='D')])
def test_to_hdf_with_object_column_names_should_fail(tmp_path, setup_path, columns):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=columns)
    path = tmp_path / setup_path
    msg = 'cannot have non-object label DataIndexableCol'
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key='df', format='table', data_columns=True)