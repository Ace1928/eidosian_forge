import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.feather_format import read_feather, to_feather  # isort:skip
@pytest.mark.network
@pytest.mark.single_cpu
def test_http_path(self, feather_file, httpserver):
    expected = read_feather(feather_file)
    with open(feather_file, 'rb') as f:
        httpserver.serve_content(content=f.read())
        res = read_feather(httpserver.url)
    tm.assert_frame_equal(expected, res)