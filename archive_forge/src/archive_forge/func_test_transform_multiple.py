import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_multiple(ts):
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])
    grouped.transform(lambda x: x * 2)
    msg = 'using SeriesGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped.transform(np.mean)