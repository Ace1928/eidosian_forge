from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_loc_preserve_names(self, multiindex_year_month_day_dataframe_random_data):
    ymd = multiindex_year_month_day_dataframe_random_data
    result = ymd.loc[2000]
    result2 = ymd['A'].loc[2000]
    assert result.index.names == ymd.index.names[1:]
    assert result2.index.names == ymd.index.names[1:]
    result = ymd.loc[2000, 2]
    result2 = ymd['A'].loc[2000, 2]
    assert result.index.name == ymd.index.names[2]
    assert result2.index.name == ymd.index.names[2]