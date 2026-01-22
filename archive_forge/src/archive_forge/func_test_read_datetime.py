from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_read_datetime(request, engine):
    if engine == 'pyarrow':
        reason = 'Pyarrow only supports a file path as an input and line delimited json'
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))
    df = DataFrame([([1, 2], ['2020-03-05', '2020-04-08T09:58:49+00:00'], 'hector')], columns=['accounts', 'date', 'name'])
    json_line = df.to_json(lines=True, orient='records')
    if engine == 'pyarrow':
        result = read_json(StringIO(json_line), engine=engine)
    else:
        result = read_json(StringIO(json_line), engine=engine)
    expected = DataFrame([[1, '2020-03-05', 'hector'], [2, '2020-04-08T09:58:49+00:00', 'hector']], columns=['accounts', 'date', 'name'])
    tm.assert_frame_equal(result, expected)