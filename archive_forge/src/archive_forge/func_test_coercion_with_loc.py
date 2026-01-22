import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
@pytest.mark.parametrize('expected', EXPECTED_SINGLE_ROW_RESULTS)
def test_coercion_with_loc(self, expected):
    start_data, expected_result, warn = expected
    start_dataframe = DataFrame({'foo': start_data})
    start_dataframe.loc[0, ['foo']] = None
    expected_dataframe = DataFrame({'foo': expected_result})
    tm.assert_frame_equal(start_dataframe, expected_dataframe)