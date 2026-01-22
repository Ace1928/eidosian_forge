from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
@pytest.mark.parametrize('chunksize', [1, 1.0])
def test_readjson_chunks(request, lines_json_df, chunksize, engine):
    if engine == 'pyarrow':
        reason = "Pyarrow only supports a file path as an input and line delimited jsonand doesn't support chunksize parameter."
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))
    unchunked = read_json(StringIO(lines_json_df), lines=True)
    with read_json(StringIO(lines_json_df), lines=True, chunksize=chunksize, engine=engine) as reader:
        chunked = pd.concat(reader)
    tm.assert_frame_equal(chunked, unchunked)