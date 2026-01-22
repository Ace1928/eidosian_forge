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
def test_loc_setitem_consistency_slice_column_len(self):
    levels = [['Region_1'] * 4, ['Site_1', 'Site_1', 'Site_2', 'Site_2'], [3987227376, 3980680971, 3977723249, 3977723089]]
    mi = MultiIndex.from_arrays(levels, names=['Region', 'Site', 'RespondentID'])
    clevels = [['Respondent', 'Respondent', 'Respondent', 'OtherCat', 'OtherCat'], ['Something', 'StartDate', 'EndDate', 'Yes/No', 'SomethingElse']]
    cols = MultiIndex.from_arrays(clevels, names=['Level_0', 'Level_1'])
    values = [['A', '5/25/2015 10:59', '5/25/2015 11:22', 'Yes', np.nan], ['A', '5/21/2015 9:40', '5/21/2015 9:52', 'Yes', 'Yes'], ['A', '5/20/2015 8:27', '5/20/2015 8:41', 'Yes', np.nan], ['A', '5/20/2015 8:33', '5/20/2015 9:09', 'Yes', 'No']]
    df = DataFrame(values, index=mi, columns=cols)
    df.loc[:, ('Respondent', 'StartDate')] = to_datetime(df.loc[:, ('Respondent', 'StartDate')])
    df.loc[:, ('Respondent', 'EndDate')] = to_datetime(df.loc[:, ('Respondent', 'EndDate')])
    df = df.infer_objects(copy=False)
    df.loc[:, ('Respondent', 'Duration')] = df.loc[:, ('Respondent', 'EndDate')] - df.loc[:, ('Respondent', 'StartDate')]
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.loc[:, ('Respondent', 'Duration')] = df.loc[:, ('Respondent', 'Duration')] / Timedelta(60000000000)
    expected = Series([23.0, 12.0, 14.0, 36.0], index=df.index, name=('Respondent', 'Duration'))
    tm.assert_series_equal(df['Respondent', 'Duration'], expected)