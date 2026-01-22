import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
@pytest.mark.parametrize('df', [DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, np.nan]), DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1.1, 2.2, np.nan]), DataFrame([[0, 1, 2, 3], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]), DataFrame([[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]), DataFrame([[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1, 2, 2])])
def test_constructor_with_nas(self, df):
    for i in range(len(df.columns)):
        df.iloc[:, i]
    indexer = np.arange(len(df.columns))[isna(df.columns)]
    if len(indexer) == 0:
        with pytest.raises(KeyError, match='^nan$'):
            df.loc[:, np.nan]
    elif len(indexer) == 1:
        tm.assert_series_equal(df.iloc[:, indexer[0]], df.loc[:, np.nan])
    else:
        tm.assert_frame_equal(df.iloc[:, indexer], df.loc[:, np.nan])