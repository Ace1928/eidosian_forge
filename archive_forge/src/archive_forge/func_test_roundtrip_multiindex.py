import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.xfail(reason='#50456 Column multiindex is stored and loaded differently', raises=AssertionError)
@pytest.mark.parametrize('columns', [[['2022', '2022'], ['JAN', 'FEB']], [['2022', '2023'], ['JAN', 'JAN']], [['2022', '2022'], ['JAN', 'JAN']]])
def test_roundtrip_multiindex(self, columns):
    df = DataFrame([[1, 2], [3, 4]], columns=pd.MultiIndex.from_arrays(columns))
    data = StringIO(df.to_json(orient='split'))
    result = read_json(data, orient='split')
    tm.assert_frame_equal(result, df)