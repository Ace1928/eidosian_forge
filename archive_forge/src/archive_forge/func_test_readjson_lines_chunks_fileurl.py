from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_readjson_lines_chunks_fileurl(request, datapath, engine):
    if engine == 'pyarrow':
        reason = "Pyarrow only supports a file path as an input and line delimited jsonand doesn't support chunksize parameter."
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))
    df_list_expected = [DataFrame([[1, 2]], columns=['a', 'b'], index=[0]), DataFrame([[3, 4]], columns=['a', 'b'], index=[1]), DataFrame([[5, 6]], columns=['a', 'b'], index=[2])]
    os_path = datapath('io', 'json', 'data', 'line_delimited.json')
    file_url = Path(os_path).as_uri()
    with read_json(file_url, lines=True, chunksize=1, engine=engine) as url_reader:
        for index, chuck in enumerate(url_reader):
            tm.assert_frame_equal(chuck, df_list_expected[index])